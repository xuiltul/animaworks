#!/usr/bin/env python3
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Render the macOS files for one AnimaWorks enclave instance from a spec.

One isolated runtime per trust boundary: every enclave on a host gets its own
OS user and group, data directory, socket directory, port, LaunchDaemons and
pf anchor.  This script writes everything an operator installs for one of
them, so adding an enclave means writing one spec file instead of editing
templates by hand::

    python3 scripts/enclave/render_instance.py --spec isms.json --out build/isms
    sudo sh build/isms/install.sh prepare

``--check`` compares the rendered LaunchDaemons and boot-prep script with the
installed ones instead of writing anything, e.g. to confirm that a spec
describes an enclave that is already running.

The spec is JSON.  Required keys: ``name``, ``user``, ``uid``, ``group``,
``gid``, ``host_user`` and ``port``.  The rest derive from those; the optional
``label``, ``base_dir``, ``socket_name``, ``boot_prep_script`` and
``egress_anchor_name`` let a spec describe an enclave installed before this
script without moving it (``egress_anchor_name: ""`` keeps the
single-instance pf anchor).  ``templates/_shared/enclave/instance.example.json``
lists every key.
"""

from __future__ import annotations

import argparse
import difflib
import ipaddress
import json
import plistlib
import pwd
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO / "templates" / "_shared" / "enclave"
EGRESS_SCRIPT = REPO / "scripts" / "enclave" / "egress_firewall.py"
LAUNCH_DAEMONS = Path("/Library/LaunchDaemons")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_ROLE_USER_RE = re.compile(r"^_[a-z][a-z0-9_]{1,30}$")
_ACCOUNT_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PORTS_RE = re.compile(r"^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$")
_HOST_PORT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*:[0-9]{1,5}$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")

# Role accounts (names starting with "_") take a uid/gid in this range.
_ROLE_ID_RANGE = range(200, 401)


class SpecError(ValueError):
    """Raised when a spec is missing a key or holds an unsafe value."""


@dataclass(frozen=True)
class Instance:
    """Every value the rendered files need, validated and with defaults applied."""

    name: str
    user: str
    uid: int
    group: str
    gid: int
    host_user: str
    host_uid: int
    port: int
    home: str
    label: str
    base_dir: str
    socket_name: str
    boot_prep_script: str
    egress_script: str
    egress_anchor_name: str
    region: str
    loopback_ports: str
    allow_endpoints: list[str]
    allow_hosts: list[str]
    lang: str
    environment: dict[str, str]
    uv: str
    entry_anima: str
    allowed_animas: list[str]
    llm_credentials: list[str]
    host_api_port: int
    egress_stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def data_dir(self) -> str:
        return f"{self.home}/.animaworks"

    @property
    def code_dir(self) -> str:
        return f"{self.home}/animaworks"

    @property
    def secrets_dir(self) -> str:
        return f"{self.data_dir}/secrets"

    @property
    def run_dir(self) -> str:
        return f"{self.base_dir}/run"

    @property
    def socket_path(self) -> str:
        return f"{self.run_dir}/{self.socket_name}"

    @property
    def egress_label(self) -> str:
        return f"{self.label}-egress"

    @property
    def boot_prep_label(self) -> str:
        return f"{self.label}-boot-prep"

    @property
    def log_stem(self) -> str:
        # com.animaworks.enclave.isms -> animaworks-enclave-isms
        return self.label.removeprefix("com.").replace(".", "-")

    def template_values(self) -> dict[str, str]:
        return {
            "name": self.name,
            "user": self.user,
            "uid": str(self.uid),
            "group": self.group,
            "gid": str(self.gid),
            "host_user": self.host_user,
            "home": self.home,
            "data_dir": self.data_dir,
            "code_dir": self.code_dir,
            "secrets_dir": self.secrets_dir,
            "base_dir": self.base_dir,
            "run_dir": self.run_dir,
            "label": self.label,
            "egress_label": self.egress_label,
            "boot_prep_label": self.boot_prep_label,
            "boot_prep_script": self.boot_prep_script,
            "egress_script": self.egress_script,
            "uv": self.uv,
        }


def _require(spec: dict[str, Any], key: str) -> Any:
    if key not in spec:
        raise SpecError(f"spec is missing '{key}'")
    return spec[key]


def _match(pattern: re.Pattern[str], value: Any, key: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise SpecError(f"'{key}' has an unsafe or malformed value")
    return value


def _path(value: Any, key: str) -> str:
    path = _match(_PATH_RE, value, key)
    if ".." in path.split("/"):
        raise SpecError(f"'{key}' must not contain '..'")
    return path.rstrip("/") or "/"


def _int_in(value: Any, key: str, allowed: range) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        raise SpecError(f"'{key}' must be an integer in {allowed.start}-{allowed.stop - 1}")
    return value


def _str_list(value: Any, key: str, pattern: re.Pattern[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecError(f"'{key}' must be a list of strings")
    if pattern is not None:
        for item in value:
            _match(pattern, item, key)
    return list(value)


def _endpoints(value: Any) -> list[str]:
    endpoints = _str_list(value, "egress.allow_endpoints")
    for item in endpoints:
        host, _, port = item.rpartition(":")
        try:
            ipaddress.ip_address(host.strip("[]"))
        except ValueError as exc:
            raise SpecError("'egress.allow_endpoints' takes IP:PORT; put names in 'egress.allow_hosts'") from exc
        _int_in(int(port) if port.isdigit() else -1, "egress.allow_endpoints port", range(1, 65536))
    return endpoints


def _host_uid(spec: dict[str, Any], host_user: str) -> int:
    if "host_uid" in spec:
        return _int_in(spec["host_uid"], "host_uid", range(0, 2**31))
    try:
        return pwd.getpwnam(host_user).pw_uid
    except KeyError as exc:
        raise SpecError(f"host user '{host_user}' does not exist here; set 'host_uid'") from exc


def load_instance(spec: dict[str, Any]) -> Instance:
    """Validate *spec* and fill in the derived values.

    Every value ends up in shell scripts or LaunchDaemons as-is, so values are
    checked against strict patterns rather than quoted.
    """
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    name = _match(_NAME_RE, _require(spec, "name"), "name")
    user = _match(_ROLE_USER_RE, _require(spec, "user"), "user")
    group = _match(_ACCOUNT_RE, _require(spec, "group"), "group")
    host_user = _match(_ACCOUNT_RE, _require(spec, "host_user"), "host_user")
    egress = spec.get("egress", {})
    if not isinstance(egress, dict):
        raise SpecError("'egress' must be an object")
    port = _int_in(_require(spec, "port"), "port", range(1024, 65536))
    region = egress.get("region", "")
    if region:
        _match(re.compile(r"^[a-z]{2}(-[a-z]+)+-[0-9]$"), region, "egress.region")
    environment = spec.get("environment", {})
    if not isinstance(environment, dict):
        raise SpecError("'environment' must be an object")
    for key, value in environment.items():
        _match(_ENV_NAME_RE, key, "environment")
        if not isinstance(value, str) or "\n" in value:
            raise SpecError("'environment' values must be single-line strings")
    stages = spec.get("egress_stages", [{"type": "masker", "profile": "default"}])
    if not isinstance(stages, list) or not stages:
        raise SpecError("'egress_stages' must be a non-empty list")
    anchor_name = spec.get("egress_anchor_name", name)
    if anchor_name:
        _match(_NAME_RE, anchor_name, "egress_anchor_name")
    return Instance(
        name=name,
        user=user,
        uid=_int_in(_require(spec, "uid"), "uid", _ROLE_ID_RANGE),
        group=group,
        gid=_int_in(_require(spec, "gid"), "gid", _ROLE_ID_RANGE),
        host_user=host_user,
        host_uid=_host_uid(spec, host_user),
        port=port,
        home=_path(spec.get("home", f"/Users/{user}"), "home"),
        label=_match(_LABEL_RE, spec.get("label", f"com.animaworks.enclave.{name}"), "label"),
        base_dir=_path(spec.get("base_dir", f"/usr/local/var/animaworks-enclave-{name}"), "base_dir"),
        socket_name=_match(_LABEL_RE, spec.get("socket_name", f"{name}.sock"), "socket_name"),
        boot_prep_script=_path(
            spec.get("boot_prep_script", f"/usr/local/libexec/animaworks-enclave-{name}-boot-prep.sh"),
            "boot_prep_script",
        ),
        egress_script=_path(
            spec.get("egress_script", "/usr/local/libexec/animaworks-enclave-egress.py"), "egress_script"
        ),
        egress_anchor_name=anchor_name,
        region=region,
        loopback_ports=_match(_PORTS_RE, str(egress.get("loopback_ports", port)), "egress.loopback_ports"),
        allow_endpoints=_endpoints(egress.get("allow_endpoints", [])),
        allow_hosts=_str_list(egress.get("allow_hosts", []), "egress.allow_hosts", _HOST_PORT_RE),
        lang=_match(re.compile(r"^[A-Za-z_]+(\.[A-Za-z0-9-]+)?$"), spec.get("lang", "ja_JP.UTF-8"), "lang"),
        environment=dict(environment),
        uv=_path(spec.get("uv", "/opt/homebrew/bin/uv"), "uv"),
        entry_anima=_match(re.compile(r"^[a-z0-9_-]*$"), spec.get("entry_anima", ""), "entry_anima"),
        allowed_animas=_str_list(spec.get("allowed_animas", []), "allowed_animas", re.compile(r"^[a-z0-9_-]+$")),
        llm_credentials=_str_list(
            spec.get("llm_credentials", ["anthropic"]), "llm_credentials", re.compile(r"^[A-Za-z0-9_.-]+$")
        ),
        host_api_port=_int_in(spec.get("host_api_port", 18500), "host_api_port", range(1, 65536)),
        egress_stages=stages,
    )


# ── LaunchDaemons ───────────────────────────────────────────


def instance_daemon(inst: Instance) -> dict[str, Any]:
    env = {
        "HOME": inst.home,
        "ANIMAWORKS_DATA_DIR": inst.data_dir,
        # Mode A animas find animaworks-tool here (install.sh links it).
        "PATH": f"{inst.home}/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
        "LANG": inst.lang,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "AWS_STS_REGIONAL_ENDPOINTS": "regional",
        **inst.environment,
    }
    return {
        "Label": inst.label,
        "UserName": inst.user,
        "GroupName": inst.group,
        "InitGroups": True,
        "Umask": 0o077,
        "EnvironmentVariables": env,
        "ProgramArguments": [
            f"{inst.code_dir}/.venv/bin/animaworks",
            "start",
            "--foreground",
            "--host",
            "127.0.0.1",
            "--port",
            str(inst.port),
        ],
        "WorkingDirectory": inst.home,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "StandardOutPath": f"{inst.data_dir}/logs/launchd.out.log",
        "StandardErrorPath": f"{inst.data_dir}/logs/launchd.err.log",
    }


def egress_daemon(inst: Instance) -> dict[str, Any]:
    args = ["/usr/bin/python3", inst.egress_script, "--backend", "pf", "--user", inst.user]
    if inst.egress_anchor_name:
        args += ["--name", inst.egress_anchor_name]
    if inst.region:
        args += ["--region", inst.region]
    args += ["--loopback-ports", inst.loopback_ports]
    if inst.allow_endpoints:
        args += ["--allow-endpoints", ",".join(inst.allow_endpoints)]
    if inst.allow_hosts:
        args += ["--allow-hosts", ",".join(inst.allow_hosts)]
    args.append("--apply")
    log = f"/var/log/{inst.log_stem}-egress.log"
    return {
        "Label": inst.egress_label,
        "ProgramArguments": args,
        "RunAtLoad": True,
        "StartInterval": 3600,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }


def boot_prep_daemon(inst: Instance) -> dict[str, Any]:
    log = f"/var/log/{inst.log_stem}-boot-prep.log"
    return {
        "Label": inst.boot_prep_label,
        "ProgramArguments": [inst.boot_prep_script],
        "RunAtLoad": True,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }


def daemons(inst: Instance) -> dict[str, dict[str, Any]]:
    return {d["Label"]: d for d in (instance_daemon(inst), egress_daemon(inst), boot_prep_daemon(inst))}


# ── Scripts and config fragments ────────────────────────────


def render_template(name: str, values: dict[str, str]) -> str:
    text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise SpecError(f"template {name} uses unknown value '{key}'")
        return values[key]

    return _PLACEHOLDER_RE.sub(_sub, text)


def enclave_config(inst: Instance) -> dict[str, Any]:
    """The instance's ``config.json`` keys (merge into the generated config)."""
    return {
        "enclave": {
            "enabled": True,
            "name": inst.name,
            "socket_path": inst.socket_path,
            "socket_group": inst.group,
            "entry_anima": inst.entry_anima,
            "allowed_peer_uids": [inst.host_uid],
            "allowed_llm_credentials": inst.llm_credentials,
            "secrets_dir": inst.secrets_dir,
            "raw_dir": "raw",
            "egress": {"stages": inst.egress_stages},
            # The startup guard refuses to boot while these are reachable,
            # i.e. before the pf rules are loaded.
            "egress_probe": [f"127.0.0.1:{inst.host_api_port}", "1.1.1.1:443"],
        },
        "external_tasks": {"enabled": False},
    }


def host_config(inst: Instance) -> dict[str, Any]:
    """The host's ``config.json`` keys that let its animas ask this enclave."""
    return {
        "enclaves": {
            inst.name: {
                "socket_path": inst.socket_path,
                "allowed_animas": inst.allowed_animas,
                "timeout_s": 900,
            }
        }
    }


def render(inst: Instance) -> dict[str, bytes]:
    """Return every rendered file as ``{relative path: content}``."""
    values = inst.template_values()
    files: dict[str, bytes] = {
        f"{label}.plist": plistlib.dumps(daemon, sort_keys=False) for label, daemon in daemons(inst).items()
    }
    files["boot-prep.sh"] = render_template("boot-prep.sh.tmpl", values).encode()
    files["install.sh"] = render_template("install.sh.tmpl", values).encode()
    files["egress_firewall.py"] = EGRESS_SCRIPT.read_bytes()
    files["enclave-config.json"] = (json.dumps(enclave_config(inst), indent=2, ensure_ascii=False) + "\n").encode()
    files["host-config.json"] = (json.dumps(host_config(inst), indent=2, ensure_ascii=False) + "\n").encode()
    return files


def write(files: dict[str, bytes], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = out / rel
        path.write_bytes(content)
        path.chmod(0o755 if rel.endswith((".sh", ".py")) else 0o644)


# ── Drift check ─────────────────────────────────────────────


def _script_body(text: str) -> list[str]:
    """Lines that run: comments and blank lines are not compared."""
    return [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def check(inst: Instance, daemon_dir: Path = LAUNCH_DAEMONS) -> list[str]:
    """Return differences between the rendered and the installed files (empty when they match)."""
    problems: list[str] = []
    for label, rendered in daemons(inst).items():
        installed_path = daemon_dir / f"{label}.plist"
        try:
            with installed_path.open("rb") as stream:
                installed = plistlib.load(stream)
        except OSError:
            problems.append(f"{installed_path}: not installed")
            continue
        for key in sorted(set(rendered) | set(installed)):
            if rendered.get(key) != installed.get(key):
                problems.append(
                    f"{installed_path}: {key} differs (installed {installed.get(key)!r}, rendered {rendered.get(key)!r})"
                )
    script = Path(inst.boot_prep_script)
    try:
        installed_script = script.read_text(encoding="utf-8")
    except OSError:
        problems.append(f"{script}: not installed")
    else:
        rendered_script = render_template("boot-prep.sh.tmpl", inst.template_values())
        diff = list(
            difflib.unified_diff(
                _script_body(installed_script), _script_body(rendered_script), "installed", "rendered", lineterm=""
            )
        )
        if diff:
            problems.append(f"{script}: commands differ\n" + "\n".join(diff))
    return problems


def _next_steps(inst: Instance, out: Path) -> str:
    return f"""Wrote {out}.  Next, as root unless noted:
  1. sh {out}/install.sh prepare
  2. git bundle create /tmp/animaworks.bundle HEAD   (in the AnimaWorks checkout, as the host user)
     sh {out}/install.sh code /tmp/animaworks.bundle
  3. as {inst.user}: run `animaworks init`, merge {out}/enclave-config.json into {inst.data_dir}/config.json,
     set auth.json to password mode with trust_localhost=false, create the animas
     (permissions.json file_roots must not be "/") and put secrets in {inst.secrets_dir}
  4. sh {out}/install.sh daemons
  5. sh {out}/install.sh start
  6. as {inst.host_user}: merge {out}/host-config.json into the host config.json and restart the host
     (it joined {inst.group} in step 1); then `animaworks enclave doctor` on both sides
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path, help="instance spec (JSON)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="directory to write the rendered files into")
    action.add_argument("--check", action="store_true", help="compare with the installed files instead")
    args = parser.parse_args(argv)

    try:
        inst = load_instance(json.loads(args.spec.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, SpecError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if args.check:
        problems = check(inst)
        for problem in problems:
            sys.stdout.write(problem + "\n")
        if not problems:
            sys.stdout.write(f"{inst.label}: installed files match the spec\n")
        return 1 if problems else 0

    write(render(inst), args.out)
    sys.stdout.write(_next_steps(inst, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
