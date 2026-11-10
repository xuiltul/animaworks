"""A fake 1Password CLI used by the vault unit tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT = """#!__PYTHON__
from __future__ import annotations

import json
import os
import sys


def _option(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def main() -> int:
    state_path = os.environ["FAKE_OP_STATE"]
    argv_path = os.environ["FAKE_OP_ARGV"]
    state = json.loads(open(state_path, encoding="utf-8").read())
    argv = sys.argv[1:]
    with open(argv_path, "a", encoding="utf-8") as log:
        log.write(json.dumps(argv) + "\\n")

    if argv[:1] == ["read"]:
        ref = argv[-1]
        if ref not in state.get("values", {}):
            print("item not found", file=sys.stderr)
            return 1
        sys.stdout.write(state["values"][ref])
        return 0

    if argv[:2] == ["item", "get"]:
        title = argv[2]
        vault = _option(argv, "--vault")
        item_id = state.get("items", {}).get(vault, {}).get(title)
        if item_id is None:
            print("item not found", file=sys.stderr)
            return 1
        sys.stdout.write(json.dumps({"id": item_id}))
        return 0

    if argv[:2] == ["item", "create"]:
        vault = _option(argv, "--vault")
        template = json.loads(sys.stdin.read())
        item_id = state.get("create_id", "item-created-1")
        state.setdefault("items", {}).setdefault(vault, {})[template["title"]] = item_id
        state.setdefault("created", []).append({"vault": vault, "template": template})
        open(state_path, "w", encoding="utf-8").write(json.dumps(state))
        sys.stdout.write(json.dumps({"id": item_id}))
        return 0

    print("unsupported fake op invocation", file=sys.stderr)
    return 2


sys.exit(main())
"""


class FakeOp:
    """Create an executable fake ``op`` script backed by a JSON state file."""

    def __init__(self, tmp_path: Path) -> None:
        self.script_path = tmp_path / "fake_op.py"
        self.state_path = tmp_path / "fake_op_state.json"
        self.argv_path = tmp_path / "fake_op_argv.log"
        self.script_path.write_text(_SCRIPT.replace("__PYTHON__", sys.executable), encoding="utf-8")
        self.script_path.chmod(0o755)
        self.argv_path.write_text("", encoding="utf-8")
        self.write_state({"values": {}, "items": {}, "created": [], "create_id": "item-created-1"})

    @property
    def command(self) -> list[str]:
        return [str(self.script_path)]

    @property
    def env(self) -> dict[str, str]:
        return {"FAKE_OP_STATE": str(self.state_path), "FAKE_OP_ARGV": str(self.argv_path)}

    def read_state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def set_value(self, ref: str, value: str) -> None:
        state = self.read_state()
        state["values"][ref] = value
        self.write_state(state)

    def remove_value(self, ref: str) -> None:
        state = self.read_state()
        state["values"].pop(ref, None)
        self.write_state(state)

    def set_item(self, vault: str, title: str, item_id: str) -> None:
        state = self.read_state()
        state["items"].setdefault(vault, {})[title] = item_id
        self.write_state(state)

    def set_create_id(self, item_id: str) -> None:
        state = self.read_state()
        state["create_id"] = item_id
        self.write_state(state)

    def argv_log(self) -> list[list[str]]:
        return [json.loads(line) for line in self.argv_path.read_text(encoding="utf-8").splitlines() if line]

    def created(self) -> list[dict]:
        return self.read_state().get("created", [])
