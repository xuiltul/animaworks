#!/usr/bin/env python3
"""Debug IPC streaming: connect directly to anima Unix socket and log every response."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid


IPC_BUFFER_LIMIT = 16 * 1024 * 1024


async def main(anima_name: str, message: str) -> None:
    from core.paths import get_data_dir

    socket_path = str(get_data_dir() / "run" / "sockets" / f"{anima_name}.sock")
    print(f"=== Direct IPC Debug: {anima_name} ===")
    print(f"Socket: {socket_path}")
    print(f"Message: {message}")
    print(f"{'='*70}")
    print()

    reader, writer = await asyncio.open_unix_connection(
        path=socket_path,
        limit=IPC_BUFFER_LIMIT,
    )

    req_id = str(uuid.uuid4())[:8]
    request = {
        "id": req_id,
        "method": "process_message",
        "params": {
            "message": message,
            "from_person": "debug_user",
            "intent": "",
            "stream": True,
            "images": None,
            "attachment_paths": None,
            "thread_id": "debug_ipc_test",
        },
    }

    writer.write((json.dumps(request) + "\n").encode())
    await writer.drain()

    t0 = time.monotonic()
    prev_t = t0
    chunk_count = 0
    text_events = 0

    while True:
        try:
            line_bytes = await asyncio.wait_for(reader.readline(), timeout=300.0)
        except asyncio.TimeoutError:
            print("  [TIMEOUT]")
            break

        if not line_bytes:
            print("  [CONNECTION CLOSED]")
            break

        now = time.monotonic()
        elapsed = now - t0
        gap = now - prev_t
        chunk_count += 1

        line = line_bytes.decode().strip()
        if not line:
            continue

        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  INVALID JSON: {line[:200]}")
            prev_t = now
            continue

        is_stream = resp.get("stream", False)
        is_done = resp.get("done", False)
        chunk_data = resp.get("chunk")
        result_data = resp.get("result")
        error_data = resp.get("error")

        if chunk_data:
            try:
                parsed = json.loads(chunk_data)
                event_type = parsed.get("type", "?")

                if event_type == "text_delta":
                    text = parsed.get("text", "")
                    text_events += 1
                    print(
                        f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                        f"IPC#{chunk_count:4d} stream chunk: text_delta  "
                        f"len={len(text):4d}  {repr(text[:100])}"
                    )
                elif event_type == "keepalive":
                    print(
                        f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                        f"IPC#{chunk_count:4d} keepalive"
                    )
                elif event_type in ("thinking_delta", "thinking_start", "thinking_end"):
                    text = parsed.get("text", "")
                    print(
                        f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                        f"IPC#{chunk_count:4d} {event_type} len={len(text)}"
                    )
                else:
                    detail = json.dumps(parsed, ensure_ascii=False)[:150]
                    print(
                        f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                        f"IPC#{chunk_count:4d} stream chunk: {event_type}  {detail}"
                    )
            except json.JSONDecodeError:
                print(
                    f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                    f"IPC#{chunk_count:4d} raw chunk: {chunk_data[:200]}"
                )
        elif is_done:
            resp_text = ""
            if result_data:
                resp_text = result_data.get("response", "")[:100]
            print(
                f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                f"IPC#{chunk_count:4d} DONE  response_len={len(result_data.get('response','') if result_data else '')}"
            )
            break
        elif error_data:
            print(
                f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                f"IPC#{chunk_count:4d} ERROR  {error_data}"
            )
            break
        else:
            print(
                f"  [{elapsed:7.3f}s] +{gap*1000:7.1f}ms  "
                f"IPC#{chunk_count:4d} OTHER  {json.dumps(resp, ensure_ascii=False)[:200]}"
            )

        prev_t = now

    writer.close()
    await writer.wait_closed()

    total = time.monotonic() - t0
    print()
    print(f"{'='*70}")
    print(f"Total IPC chunks: {chunk_count}")
    print(f"text_delta events: {text_events}")
    print(f"Total elapsed: {total:.3f}s")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kaede"
    msg = sys.argv[2] if len(sys.argv) > 2 else "こんにちは、今日の調子はどうですか？短く返答してください。"
    asyncio.run(main(name, msg))
