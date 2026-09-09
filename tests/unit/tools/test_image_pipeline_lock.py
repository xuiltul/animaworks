from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


def _generate_shared_asset(directory: str, ready, results) -> None:
    from core.config.models import ImageGenConfig
    from core.tools._image_pipeline import ImageGenPipeline

    root = Path(directory)

    def generate(**kwargs):
        with (root / "generation_calls.txt").open("a", encoding="utf-8") as output:
            output.write("generated\n")
        # Keep the first generation open long enough for the second caller
        # to encounter the same missing output after the shared barrier.
        time.sleep(0.2)
        return b"generated-image"

    client = MagicMock()
    client.generate_fullbody.side_effect = generate
    pipe = ImageGenPipeline(root, config=ImageGenConfig(image_style="realistic"))
    with patch("core.tools.image_gen._build_fullbody_client", return_value=client):
        ready.wait(timeout=15)
        result = pipe.generate_all("portrait", steps=["fullbody"])
        results.put((result.errors, result.fullbody_path is not None))


def test_background_and_reconciliation_do_not_generate_same_asset_twice(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    results = context.Queue()
    processes = [context.Process(target=_generate_shared_asset, args=(str(tmp_path), ready, results)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=25)
            assert not process.is_alive(), "pipeline process did not finish"
            assert process.exitcode == 0
        assert (tmp_path / "generation_calls.txt").read_text().splitlines() == ["generated"]
        assert results.get(timeout=2) == ([], True)
        assert results.get(timeout=2) == ([], True)
        assert (tmp_path / "assets/avatar_fullbody_realistic.png").read_bytes() == b"generated-image"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()
