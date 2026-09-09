"""Root-owned vector memory service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from core.memory.rag.store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class MemoryServiceUnavailable(RuntimeError):
    """The root memory service cannot safely answer a request."""


class MemoryService:
    """Serialize one anima's native vector operations on one bounded worker."""

    def __init__(
        self,
        anima_name: str,
        anima_dir: Path,
        *,
        queue_limit: int = 64,
        opener: Callable[[], VectorStore] | None = None,
        repair_fenced: Callable[[], bool] | None = None,
    ) -> None:
        if queue_limit < 1:
            raise ValueError("queue_limit must be >= 1")
        self.anima_name = anima_name
        self.anima_dir = anima_dir
        self.queue_limit = queue_limit
        self._opener = opener or self._open_native_store
        self._repair_fenced = repair_fenced or (lambda: False)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"memory-{anima_name}")
        self._store: VectorStore | None = None
        self._open_error: Exception | None = None
        self._pending = 0
        self._started = False
        self._closing = False
        self._repairing = False
        self._repair_lock = asyncio.Lock()

    async def start(self) -> None:
        """Open Chroma off the root event loop; failure leaves root available."""
        if self._started:
            return
        self._started = True
        try:
            self._store = await asyncio.get_running_loop().run_in_executor(self._executor, self._opener)
        except Exception as exc:
            self._open_error = exc
            logger.warning("Root memory store open failed for %s: %s", self.anima_name, exc)
            try:
                self._request_startup_repair(exc)
            except Exception:
                logger.warning("Failed to mark background RAG repair for %s", self.anima_name, exc_info=True)

    async def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run one checked operation or raise an explicit unavailable error."""
        if self._closing:
            raise MemoryServiceUnavailable("memory service is closing")
        if not self._started:
            await self.start()
        if self._closing:
            raise MemoryServiceUnavailable("memory service is closing")
        if self._repairing:
            raise MemoryServiceUnavailable("RAG repair in progress")
        if self._store is None:
            raise MemoryServiceUnavailable(f"memory store unavailable: {self._open_error or 'not open'}")
        try:
            fenced = self._repair_fenced()
        except Exception as exc:
            raise MemoryServiceUnavailable(f"repair fence state unavailable: {exc}") from exc
        if fenced:
            raise MemoryServiceUnavailable("RAG repair in progress")
        if self._pending >= self.queue_limit:
            raise MemoryServiceUnavailable("memory queue is full")
        if method == "memory.apply_access_updates":
            self._validate_access_updates(params)

        self._pending += 1
        queued_at = perf_counter()
        if method == "memory.apply_access_updates":
            future = asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._dispatch_timed,
                method,
                params,
                queued_at,
            )
            future.add_done_callback(self._deferred_operation_done)
            return {"ok": True}
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._dispatch_timed,
                method,
                params,
                queued_at,
            )
        except MemoryServiceUnavailable:
            raise
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("Root memory operation failed for %s: %s", self.anima_name, exc)
            raise MemoryServiceUnavailable(f"memory operation failed: {exc}") from exc
        finally:
            self._pending -= 1

    def _deferred_operation_done(self, future: asyncio.Future) -> None:
        self._pending -= 1
        try:
            future.result()
        except Exception:
            logger.warning("Deferred root memory operation failed for %s", self.anima_name, exc_info=True)

    def _dispatch_timed(self, method: str, params: dict[str, Any], queued_at: float) -> dict[str, Any]:
        started = perf_counter()
        try:
            return self._dispatch(method, params)
        finally:
            if method in {"memory.query", "memory.apply_access_updates"}:
                logger.info(
                    "Root memory operation: anima=%s method=%s queue_wait=%.3fs execute=%.3fs",
                    self.anima_name,
                    method,
                    started - queued_at,
                    perf_counter() - started,
                )

    async def repair(self, *, include_shared: bool) -> dict[str, Any]:
        """Rebuild, swap, reopen, and verify this root's sole vector store."""
        if not include_shared:
            from core.i18n import t

            # This operation replaces the whole DB, not just personal
            # collections. A partial rebuild would discard shared memories.
            raise ValueError(t("rag.phase3_repair_requires_shared"))
        if self._repair_lock.locked():
            raise MemoryServiceUnavailable("RAG repair already in progress")
        async with self._repair_lock:
            self._repairing = True
            try:
                staging, chunks, shared_hashes = await self._build_staging_subprocess(include_shared)
                return await self._promote_and_verify(staging, chunks, shared_hashes)
            finally:
                self._repairing = False

    async def _build_staging_subprocess(self, include_shared: bool) -> tuple[Path, int, dict[str, str]]:
        token = uuid.uuid4().hex
        staging = self.anima_dir / f"vectordb.staging-root-{token}"
        result_path = self.anima_dir / "state" / f".rag-repair-result-{token}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "core.memory.rag.repair_rebuild",
            "--build-staging",
            "--anima",
            self.anima_name,
            "--anima-dir",
            str(self.anima_dir),
            "--staging",
            str(staging),
            "--result",
            str(result_path),
        ]
        if include_shared:
            cmd.append("--shared")
        from core.config import load_config

        timeout = int(getattr(load_config().rag, "repair_timeout_seconds", 1800))
        # Staging build legitimately opens native chroma (in the staging dir),
        # so grant direct access explicitly — the process-local grant of this
        # root process does not propagate through the environment.
        from core.memory.rag.direct_access import DIRECT_CHROMA_ENV

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, DIRECT_CHROMA_ENV: "1"},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            try:
                _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RuntimeError(f"root RAG staging rebuild timed out after {timeout}s") from None
            if proc.returncode != 0:
                detail = stderr[-4000:].decode(errors="replace").strip()
                raise RuntimeError(detail or f"staging rebuild exited with code {proc.returncode}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            chunks = payload.get("chunks")
            shared_hashes = payload.get("shared_hashes")
            if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 0:
                raise RuntimeError("staging rebuild returned an invalid chunk count")
            if not isinstance(shared_hashes, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in shared_hashes.items()
            ):
                raise RuntimeError("staging rebuild returned invalid shared hashes")
            return staging, chunks, shared_hashes
        except BaseException:
            if proc.returncode is None:
                proc.kill()
                await proc.communicate()
            await asyncio.to_thread(shutil.rmtree, staging, ignore_errors=True)
            raise
        finally:
            result_path.unlink(missing_ok=True)

    async def _promote_and_verify(
        self,
        staging: Path,
        chunks: int,
        shared_hashes: dict[str, str],
    ) -> dict[str, Any]:
        """Keep promotion consistent before releasing a cancelled caller's fence.

        Executor futures cannot stop a running filesystem operation. Shield the
        entire transaction, not just individual awaits whose return values
        determine whether rollback is required.
        """
        operation = asyncio.create_task(self._promote_and_verify_consistent(staging, chunks, shared_hashes))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Repeated caller cancellation must not abandon the native worker
            # or drop repair()'s fence early. A cancelled child is already done
            # and must not be awaited forever.
            while not operation.done():
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not operation.cancelled():
                try:
                    operation.result()
                except Exception:
                    logger.warning(
                        "RAG promotion failed while caller was cancelled: %s", self.anima_name, exc_info=True
                    )
            raise

    async def _promote_and_verify_consistent(
        self,
        staging: Path,
        chunks: int,
        shared_hashes: dict[str, str],
    ) -> dict[str, Any]:
        from core.memory.rag.repair_rebuild import _backup_repair_metadata, _restore_repair_metadata
        from core.memory.rag.repair_snapshot import publish_rebuild_metadata, validate_rebuild_sources

        loop = asyncio.get_running_loop()
        backup_dir: Path | None = None
        existing: set[str] = set()
        archive: Path | None = None
        closed = False
        promoted = False
        rollback_incomplete = False
        try:
            await loop.run_in_executor(self._executor, validate_rebuild_sources, staging, self.anima_dir)
            backup_dir, existing = await loop.run_in_executor(self._executor, _backup_repair_metadata, self.anima_dir)
            closed = True
            await loop.run_in_executor(self._executor, self._close_store_sync)
            archive = await loop.run_in_executor(self._executor, self._promote_staging_sync, staging)
            promoted = True
            await loop.run_in_executor(self._executor, self._rebuild_bm25_sync)
            store = await loop.run_in_executor(self._executor, self._opener)
            self._store = store
            self._open_error = None
            verification = await loop.run_in_executor(self._executor, self._verify_store_sync, store, chunks)
            if shared_hashes:
                await loop.run_in_executor(self._executor, self._write_shared_hashes_sync, shared_hashes)
            await loop.run_in_executor(self._executor, publish_rebuild_metadata, self.anima_dir)
            await loop.run_in_executor(self._executor, self._invalidate_shared_checks_sync)
            return {
                "ok": True,
                "status": "success",
                "chunks_indexed": chunks,
                "archive_path": str(archive) if archive is not None else None,
                "verification": verification,
            }
        except BaseException as exc:
            rollback_errors: list[str] = []
            if promoted:
                try:
                    await loop.run_in_executor(self._executor, self._close_store_sync)
                    await loop.run_in_executor(self._executor, self._rollback_sync, archive)
                    assert backup_dir is not None
                    await loop.run_in_executor(
                        self._executor,
                        _restore_repair_metadata,
                        self.anima_dir,
                        backup_dir,
                        existing,
                    )
                    self._store = await loop.run_in_executor(self._executor, self._opener)
                    self._open_error = None
                except Exception as rollback_exc:
                    self._store = None
                    self._open_error = rollback_exc
                    rollback_errors.append(str(rollback_exc))
            elif closed:
                try:
                    self._store = await loop.run_in_executor(self._executor, self._opener)
                    self._open_error = None
                except Exception as reopen_exc:
                    self._open_error = reopen_exc
                    rollback_errors.append(str(reopen_exc))
            if rollback_errors:
                rollback_incomplete = True
                raise RuntimeError(f"{exc}; rollback incomplete: {'; '.join(rollback_errors)}") from exc
            raise
        finally:
            await loop.run_in_executor(self._executor, shutil.rmtree, staging, True)
            if backup_dir is not None and not rollback_incomplete:
                await loop.run_in_executor(self._executor, shutil.rmtree, backup_dir, True)

    def _close_store_sync(self) -> None:
        store, self._store = self._store, None
        if store is not None:
            store.close()

    def _promote_staging_sync(self, staging: Path) -> Path | None:
        live = self.anima_dir / "vectordb"
        archive: Path | None = None
        if live.exists():
            archive_dir = self.anima_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive = (
                archive_dir / f"vectordb-corrupt-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
            )
            shutil.move(str(live), str(archive))
        try:
            shutil.move(str(staging), str(live))
        except BaseException:
            if archive is not None and archive.exists() and not live.exists():
                shutil.move(str(archive), str(live))
            raise
        return archive

    def _rollback_sync(self, archive: Path | None) -> None:
        live = self.anima_dir / "vectordb"
        if live.exists():
            failed_dir = self.anima_dir / "archive"
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed = (
                failed_dir
                / f"vectordb-rebuild-failed-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
            )
            shutil.move(str(live), str(failed))
        if archive is not None and archive.exists():
            shutil.move(str(archive), str(live))

    def _rebuild_bm25_sync(self) -> None:
        from core.memory.bm25 import rebuild_longterm_bm25_index

        rebuild_longterm_bm25_index(self.anima_dir)

    @staticmethod
    def _verify_store_sync(store: VectorStore, expected_chunks: int) -> dict[str, int]:
        verify = getattr(store, "verify_rebuilt_data", None)
        if not callable(verify):
            raise RuntimeError("reopened root vector store cannot verify rebuilt data")
        result = verify(expected_chunks=expected_chunks)
        if not isinstance(result, dict):
            raise RuntimeError("reopened root vector store returned invalid verification")
        return result

    def _write_shared_hashes_sync(self, shared_hashes: dict[str, str]) -> None:
        from core.memory.rag.shared_meta import write_shared_hashes

        write_shared_hashes(self.anima_dir, shared_hashes)

    def _invalidate_shared_checks_sync(self) -> None:
        from core.memory.rag.shared_check_registry import invalidate_shared_checks

        invalidate_shared_checks(self.anima_name)

    def _request_startup_repair(self, error: Exception) -> None:
        from core.memory.rag import repair_state

        current = repair_state.read_state(self.anima_name, animas_dir=self.anima_dir.parent)
        if current.get("status") in repair_state.ACTIVE_REPAIR_STATUSES:
            return
        repair_state.write_repair_request_state(
            self.anima_name,
            reason="store_init_failed",
            collection=None,
            source="phase3_root_startup",
            include_shared=True,
            animas_dir=self.anima_dir.parent,
        )
        logger.warning("Marked phase3 root RAG for background repair after open failure: %s", error)

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        store = self._store
        if store is None:
            raise MemoryServiceUnavailable("memory store unavailable")
        if method == "memory.query":
            collection = self._string(params, "collection")
            embedding = params.get("embedding")
            top_k = params.get("top_k", 10)
            filter_metadata = params.get("filter_metadata")
            if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
                raise ValueError("embedding must be a list of numbers")
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
                raise ValueError("top_k must be an integer >= 1")
            if filter_metadata is not None and not isinstance(filter_metadata, dict):
                raise ValueError("filter_metadata must be an object or null")
            query = getattr(store, "_query_once", store.query)
            return {"results": self._search_results(query(collection, embedding, top_k, filter_metadata))}
        if method == "memory.list_collections_checked":
            listing = getattr(store, "_list_collections_once", store.list_collections)
            return {"collections": list(listing())}
        if method == "memory.get_by_metadata":
            collection = self._string(params, "collection")
            where = params.get("where")
            limit = params.get("limit", 20)
            if not isinstance(where, dict):
                raise ValueError("where must be an object")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ValueError("limit must be an integer >= 1")
            get = getattr(store, "_get_by_metadata_once", store.get_by_metadata)
            return {"results": self._search_results(get(collection, where, limit))}
        if method == "memory.get_by_ids":
            collection = self._string(params, "collection")
            ids = params.get("ids")
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise ValueError("ids must be a list of strings")
            get = getattr(store, "_get_by_ids_once", store.get_by_ids)
            return {"documents": self._documents(get(collection, ids))}
        if method == "memory.create_collection":
            collection = self._string(params, "collection")
            create = getattr(store, "_create_collection_once", store.create_collection)
            return {"ok": bool(create(collection))}
        if method == "memory.delete_collection":
            collection = self._string(params, "collection")
            delete = getattr(store, "_delete_collection_once", store.delete_collection)
            return {"ok": bool(delete(collection))}
        if method == "memory.upsert":
            collection = self._string(params, "collection")
            documents = self._document_params(params.get("documents"))
            upsert = getattr(store, "_upsert_once", store.upsert)
            return {"ok": bool(upsert(collection, documents))}
        if method == "memory.delete_documents":
            collection = self._string(params, "collection")
            ids = self._strings(params.get("ids"), "ids")
            delete = getattr(store, "_delete_documents_once", store.delete_documents)
            return {"ok": bool(delete(collection, ids))}
        if method == "memory.update_metadata":
            collection = self._string(params, "collection")
            ids = self._strings(params.get("ids"), "ids")
            metadatas = params.get("metadatas")
            if not isinstance(metadatas, list) or not all(isinstance(item, dict) for item in metadatas):
                raise ValueError("metadatas must be a list of objects")
            if len(ids) != len(metadatas):
                raise ValueError("ids and metadatas must have the same length")
            update = getattr(store, "_update_metadata_once", store.update_metadata)
            return {"ok": bool(update(collection, ids, metadatas))}
        if method == "memory.apply_access_updates":
            return {"ok": self._apply_access_updates(store, params["operations"])}
        raise ValueError(f"unsupported memory method: {method}")

    @staticmethod
    def _validate_access_updates(params: dict[str, Any]) -> None:
        operations = params.get("operations")
        if not isinstance(operations, list):
            raise ValueError("operations must be a list")
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("operations must contain objects")
            if not isinstance(operation.get("collection"), str) or not operation["collection"]:
                raise ValueError("collection must be a non-empty string")
            if not isinstance(operation.get("doc_id"), str) or not operation["doc_id"]:
                raise ValueError("doc_id must be a non-empty string")
            for key in ("access_delta", "retrieved_delta", "used_delta"):
                value = operation.get(key, 0)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{key} must be a non-negative number")

    @staticmethod
    def _apply_access_updates(store: VectorStore, operations: list[dict[str, Any]]) -> bool:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for operation in operations:
            grouped.setdefault(operation["collection"], []).append(operation)
        get = getattr(store, "_get_by_ids_once", store.get_by_ids)
        update = getattr(store, "_update_metadata_once", store.update_metadata)
        for collection, rows in grouped.items():
            ids = [row["doc_id"] for row in rows]
            current = {document.id: dict(document.metadata) for document in get(collection, ids)}
            metadatas: list[dict[str, str | int | float]] = []
            for row in rows:
                metadata = current.get(row["doc_id"], {})
                access_count = MemoryService._metadata_number(metadata, "access_count")
                last_accessed_at = max(
                    str(metadata.get("last_accessed_at", "") or ""),
                    str(row.get("last_accessed_at", "") or ""),
                )
                patch: dict[str, str | int | float] = {
                    "access_count": access_count + float(row.get("access_delta", 0)),
                    "retrieved_count": MemoryService._metadata_number(metadata, "retrieved_count")
                    + int(row.get("retrieved_delta", 0)),
                    "used_count": MemoryService._metadata_number(metadata, "used_count", default=access_count)
                    + int(row.get("used_delta", 0)),
                    "last_accessed_at": last_accessed_at,
                }
                for kind in ("retrieved", "used"):
                    timestamp = row.get(f"last_{kind}_at")
                    if isinstance(timestamp, str):
                        patch[f"last_{kind}_at"] = max(str(metadata.get(f"last_{kind}_at", "") or ""), timestamp)
                per_anima_key = row.get("per_anima_access_key")
                if isinstance(per_anima_key, str) and per_anima_key:
                    patch[per_anima_key] = MemoryService._metadata_number(metadata, per_anima_key) + float(
                        row.get("access_delta", 0)
                    )
                metadatas.append(patch)
            if not update(collection, ids, metadatas):
                return False
        return True

    @staticmethod
    def _metadata_number(metadata: dict[str, object], field: str, *, default: float = 0.0) -> float:
        try:
            return max(0.0, float(str(metadata.get(field, default))))
        except (TypeError, ValueError):
            return max(0.0, default)

    @staticmethod
    def _string(params: dict[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _strings(value: Any, key: str) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a list of strings")
        return value

    @staticmethod
    def _document_params(value: Any) -> list[Document]:
        if not isinstance(value, list):
            raise ValueError("documents must be a list")
        documents: list[Document] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("documents must contain objects")
            doc_id = item.get("id")
            content = item.get("content", "")
            embedding = item.get("embedding")
            metadata = item.get("metadata", {})
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("document id must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError("document content must be a string")
            if embedding is not None and (
                not isinstance(embedding, list)
                or not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in embedding)
            ):
                raise ValueError("document embedding must be a list of numbers or null")
            if not isinstance(metadata, dict):
                raise ValueError("document metadata must be an object")
            documents.append(Document(id=doc_id, content=content, embedding=embedding, metadata=metadata))
        return documents

    @staticmethod
    def _documents(documents: list[Document]) -> list[dict[str, Any]]:
        return [{"id": doc.id, "content": doc.content, "metadata": doc.metadata} for doc in documents]

    @classmethod
    def _search_results(cls, results: list[SearchResult]) -> list[dict[str, Any]]:
        return [{"document": cls._documents([result.document])[0], "score": result.score} for result in results]

    def _open_native_store(self) -> VectorStore:
        from core.memory.rag.direct_access import enable_direct_chroma_for_process
        from core.memory.rag.store import create_chroma_vector_store
        from core.paths import get_anima_vectordb_dir

        enable_direct_chroma_for_process()
        return create_chroma_vector_store(
            persist_dir=get_anima_vectordb_dir(self.anima_name),
            anima_name=self.anima_name,
        )

    async def close(self) -> None:
        """Close the sole native handle after queued operations drain."""
        if self._closing:
            return
        self._closing = True
        store = self._store
        if store is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(self._executor, store.close)
            except Exception:
                logger.warning("Failed to close root memory store for %s", self.anima_name, exc_info=True)
        self._store = None
        self._executor.shutdown(wait=False)
