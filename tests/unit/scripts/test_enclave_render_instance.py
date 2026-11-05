# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/enclave/render_instance.py (one enclave instance per spec)."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "enclave" / "render_instance.py"
_EXAMPLE = _REPO / "templates" / "_shared" / "enclave" / "instance.example.json"


@pytest.fixture
def render(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("render_instance", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the defining module through sys.modules.
    monkeypatch.setitem(sys.modules, "render_instance", module)
    spec.loader.exec_module(module)
    return module


def _spec(**overrides) -> dict:
    spec = {
        "name": "alpha",
        "user": "_awalpha",
        "uid": 320,
        "group": "animaworks-alpha",
        "gid": 320,
        "host_user": "host-service-user",
        "host_uid": 501,
        "port": 18640,
        "entry_anima": "entry-anima",
        "allowed_animas": ["host-assistant"],
        "egress": {"loopback_ports": "18640", "allow_hosts": ["llm.example.com:443"]},
    }
    spec.update(overrides)
    return spec


def test_the_example_spec_renders(render) -> None:
    inst = render.load_instance(json.loads(_EXAMPLE.read_text(encoding="utf-8")))
    files = render.render(inst)
    assert set(files) == {
        "com.animaworks.enclave.example.plist",
        "com.animaworks.enclave.example-egress.plist",
        "com.animaworks.enclave.example-boot-prep.plist",
        "boot-prep.sh",
        "install.sh",
        "egress_firewall.py",
        "enclave-config.json",
        "host-config.json",
    }


def test_new_instances_get_their_own_names_everywhere(render) -> None:
    inst = render.load_instance(_spec())
    daemons = render.daemons(inst)
    instance = daemons["com.animaworks.enclave.alpha"]
    assert instance["UserName"] == "_awalpha"
    assert instance["GroupName"] == "animaworks-alpha"
    assert instance["Umask"] == 0o077
    assert instance["ProgramArguments"][-3:] == ["127.0.0.1", "--port", "18640"]
    assert instance["EnvironmentVariables"]["ANIMAWORKS_DATA_DIR"] == "/Users/_awalpha/.animaworks"
    egress = daemons["com.animaworks.enclave.alpha-egress"]["ProgramArguments"]
    # Its own pf anchor, so loading it cannot replace another enclave's rules.
    assert egress[egress.index("--name") + 1] == "alpha"
    assert "--region" not in egress
    assert egress[-1] == "--apply"
    assert inst.socket_path == "/usr/local/var/animaworks-enclave-alpha/run/alpha.sock"
    assert daemons["com.animaworks.enclave.alpha-boot-prep"]["StandardOutPath"] == (
        "/var/log/animaworks-enclave-alpha-boot-prep.log"
    )


def test_a_spec_can_describe_a_single_instance_install(render) -> None:
    inst = render.load_instance(
        _spec(
            label="com.animaworks.enclave",
            base_dir="/usr/local/var/animaworks-enclave",
            socket_name="enclave.sock",
            boot_prep_script="/usr/local/libexec/animaworks-enclave-boot-prep.sh",
            egress_anchor_name="",
            egress={"region": "ap-northeast-1", "loopback_ports": "18640,33061-33070"},
        )
    )
    daemons = render.daemons(inst)
    assert set(daemons) == {
        "com.animaworks.enclave",
        "com.animaworks.enclave-egress",
        "com.animaworks.enclave-boot-prep",
    }
    egress = daemons["com.animaworks.enclave-egress"]
    assert "--name" not in egress["ProgramArguments"]
    assert egress["StandardOutPath"] == "/var/log/animaworks-enclave-egress.log"


def test_config_fragments_wire_host_and_enclave_together(render) -> None:
    inst = render.load_instance(_spec())
    enclave = render.enclave_config(inst)["enclave"]
    host = render.host_config(inst)["enclaves"]["alpha"]
    assert enclave["socket_path"] == host["socket_path"]
    assert enclave["socket_group"] == "animaworks-alpha"
    assert enclave["allowed_peer_uids"] == [501]
    assert enclave["secrets_dir"] == "/Users/_awalpha/.animaworks/secrets"
    assert "127.0.0.1:18500" in enclave["egress_probe"]
    assert host["allowed_animas"] == ["host-assistant"]
    assert render.enclave_config(inst)["external_tasks"] == {"enabled": False}


def test_scripts_have_every_placeholder_filled(render) -> None:
    files = render.render(render.load_instance(_spec()))
    for name in ("install.sh", "boot-prep.sh"):
        text = files[name].decode()
        assert "{{" not in text and "}}" not in text
        assert "_awalpha" in text


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_rendered_scripts_parse(render, tmp_path: Path) -> None:
    render.write(render.render(render.load_instance(_spec())), tmp_path)
    for name in ("install.sh", "boot-prep.sh"):
        subprocess.run(["sh", "-n", str(tmp_path / name)], check=True)
    assert (tmp_path / "install.sh").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "enclave-config.json").stat().st_mode & 0o777 == 0o644


def test_install_refuses_ids_owned_by_someone_else(render) -> None:
    text = render.render(render.load_instance(_spec()))["install.sh"].decode()
    assert 'fail "uid 320 is already used by another user"' in text
    assert 'fail "gid 320 is already used by another group"' in text
    assert 'fail "could not create user _awalpha with uid 320"' in text
    # A visible user with a password counts as "staff", which can read the host user's home.
    adduser = next(line for line in text.splitlines() if "sysadminctl -addUser" in line)
    assert "-roleAccount" not in adduser  # it only takes uids 450-499
    assert "dscl . -create /Users/_awalpha IsHidden 1" in text
    assert 'fail "_awalpha is still in staff"' in text
    for step in ("prepare)", "code)", "daemons)", "start)"):
        assert step in text


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("user", "awalpha"),  # role accounts start with "_"
        ("user", "_aw;rm"),
        ("uid", 501),
        ("uid", True),
        ("port", 80),
        ("home", "/Users/_awalpha/../main"),
        ("home", "/Users/_aw alpha"),
        ("label", "com.example;x"),
        ("name", "Alpha"),
        ("environment", {"lower": "x"}),
        ("environment", {"X": "a\nb"}),
        ("egress", {"allow_endpoints": ["llm.example.com:443"]}),
        ("egress", {"allow_hosts": ["llm.example.com"]}),
        ("egress", {"loopback_ports": "18640;x"}),
        ("egress_stages", []),
    ],
)
def test_unsafe_values_are_rejected(render, key: str, value) -> None:
    with pytest.raises(render.SpecError):
        render.load_instance(_spec(**{key: value}))


def test_missing_keys_are_named(render) -> None:
    spec = _spec()
    del spec["gid"]
    with pytest.raises(render.SpecError, match="gid"):
        render.load_instance(spec)


def test_check_matches_installed_files_and_reports_drift(render, tmp_path: Path) -> None:
    daemon_dir = tmp_path / "LaunchDaemons"
    daemon_dir.mkdir()
    script = tmp_path / "libexec" / "boot-prep.sh"
    inst = render.load_instance(_spec(boot_prep_script=str(script)))
    for label, daemon in render.daemons(inst).items():
        (daemon_dir / f"{label}.plist").write_bytes(plistlib.dumps(daemon))
    script.parent.mkdir()
    # Comments may differ (an install written by hand); the commands must not.
    rendered = render.render(inst)["boot-prep.sh"].decode()
    script.write_text("#!/bin/sh\n# hand-written header\nset -u" + rendered.split("set -u", 1)[1])

    assert render.check(inst, daemon_dir) == []

    egress_path = daemon_dir / "com.animaworks.enclave.alpha-egress.plist"
    egress = plistlib.loads(egress_path.read_bytes())
    egress["ProgramArguments"] = [arg for arg in egress["ProgramArguments"] if arg != "--apply"]
    egress_path.write_bytes(plistlib.dumps(egress))
    script.write_text(script.read_text().replace("-m 0750", "-m 0755", 1))

    problems = render.check(inst, daemon_dir)
    assert any("alpha-egress.plist: ProgramArguments differs" in p for p in problems)
    assert any("commands differ" in p for p in problems)


def test_check_reports_missing_installs(render, tmp_path: Path) -> None:
    inst = render.load_instance(_spec(boot_prep_script=str(tmp_path / "missing.sh")))
    problems = render.check(inst, tmp_path)
    assert sum("not installed" in p for p in problems) == 4
