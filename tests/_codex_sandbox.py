"""codex sandbox 対策: asyncio の self-pipe を os.pipe に差し替える。

codex-linux-sandbox の seccomp はネットワーク遮断のために send/sendto を
全ソケットで EPERM にするため、asyncio が self-pipe（AF_UNIX socketpair）へ
書く wake-up が黙って失敗し、to_thread / run_in_executor / call_soon_threadsafe
の完了でイベントループが起きなくなる（タイマーでしか進まない）。
os.pipe は seccomp の対象外なので、それで代用する。
"""

from __future__ import annotations

import os
import socket


def _socketpair_send_blocked() -> bool:
    try:
        a, b = socket.socketpair()
    except OSError:
        return False
    try:
        a.send(b"\0")
        return False
    except OSError:
        return True
    finally:
        a.close()
        b.close()


class _PipeEnd:
    """socket 互換の最小インターフェース（fileno/send/recv/setblocking/close）。"""

    def __init__(self, fd: int, *, writer: bool) -> None:
        self._fd = fd
        self._writer = writer

    def fileno(self) -> int:
        return self._fd

    def setblocking(self, flag: bool) -> None:
        os.set_blocking(self._fd, flag)

    def send(self, data: bytes) -> int:
        return os.write(self._fd, data)

    def recv(self, n: int) -> bytes:
        return os.read(self._fd, n)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def install() -> bool:
    if not _socketpair_send_blocked():
        return False
    # sandbox はネットワークも遮断するため、huggingface_hub がキャッシュ済みモデルの
    # 更新確認で接続失敗→バックオフ再試行を繰り返して数分止まる。オフライン固定にする。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from asyncio import selector_events

    def _make_self_pipe(self) -> None:  # type: ignore[no-untyped-def]
        r, w = os.pipe()
        self._ssock = _PipeEnd(r, writer=False)
        self._csock = _PipeEnd(w, writer=True)
        self._ssock.setblocking(False)
        self._csock.setblocking(False)
        self._internal_fds += 1
        self._add_reader(self._ssock.fileno(), self._read_from_self)

    selector_events.BaseSelectorEventLoop._make_self_pipe = _make_self_pipe  # type: ignore[method-assign]
    return True
