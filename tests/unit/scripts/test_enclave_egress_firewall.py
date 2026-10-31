from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "enclave" / "egress_firewall.py"


@pytest.fixture
def firewall(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("egress_firewall", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_aws_ranges", lambda region: (["192.0.2.0/24"], ["2001:db8::/32"]))
    return module


def test_only_the_enclave_uid_enters_the_restricting_chain(firewall) -> None:
    ruleset = firewall.build_ruleset(993, "ap-northeast-1", [18610])
    output_chain = ruleset.split("chain output {", 1)[1].split("}", 1)[0]
    # Socketless packets (FIN/ACK of closed sockets) never match skuid, so a
    # positive match keeps other users' traffic out of the reject path.
    assert "meta skuid 993 jump enclave" in output_chain
    assert "reject" not in output_chain
    assert "skuid !=" not in ruleset


def test_log_is_rate_limited_before_logging(firewall) -> None:
    ruleset = firewall.build_ruleset(993, "ap-northeast-1", [18610])
    log_line = next(line for line in ruleset.splitlines() if "log prefix" in line)
    assert log_line.strip().startswith("limit rate")
    assert ruleset.rstrip().splitlines()[-3].strip() == "reject"


def test_allowed_endpoints_are_opened_per_port(firewall) -> None:
    endpoints = firewall._parse_endpoints("192.0.2.10:8888, [2001:db8::5]:8002")
    assert endpoints == [("192.0.2.10", 8888), ("2001:db8::5", 8002)]
    ruleset = firewall.build_ruleset(993, "ap-northeast-1", [18610], endpoints)
    enclave_chain = ruleset.split("chain enclave {", 1)[1]
    assert "ip daddr 192.0.2.10 tcp dport 8888 accept" in enclave_chain
    assert "ip6 daddr 2001:db8::5 tcp dport 8002 accept" in enclave_chain
    assert enclave_chain.index("8888 accept") < enclave_chain.index("reject")


def test_endpoint_names_are_rejected(firewall) -> None:
    with pytest.raises(ValueError):
        firewall._parse_endpoints("llm.example.com:8888")


def test_allowed_hosts_resolve_to_every_address(firewall, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(name, port, proto=0):
        if name == "missing.example":
            raise OSError("no such host")
        return [(None, None, None, "", ("192.0.2.20", port)), (None, None, None, "", ("2001:db8::20", port, 0, 0))]

    monkeypatch.setattr(firewall.socket, "getaddrinfo", fake_getaddrinfo)
    endpoints = firewall._resolve_hosts("api.example:443, missing.example:443")
    assert endpoints == [("192.0.2.20", 443), ("2001:db8::20", 443)]


def test_pf_ruleset_limits_only_the_enclave_user(firewall) -> None:
    endpoints = [("192.0.2.10", 8888), ("2001:db8::5", 8002)]
    ruleset = firewall.build_pf_ruleset(299, "ap-northeast-1", [18610, 33061], endpoints)
    lines = ruleset.strip().splitlines()
    assert lines[0] == "table <aw_enclave_aws> persist { 192.0.2.0/24, 2001:db8::/32 }"
    rules = lines[1:]
    assert all(rule.endswith(("user 299", "user 299 no state")) for rule in rules)
    assert rules[0].endswith("no state") and rules[1].endswith("no state")
    assert "to any port { 53, 18610, 33061 } user 299" in rules[0]
    assert "from any port { 53, 18610, 33061 } to any user 299" in rules[1]
    assert "pass out quick proto tcp from any to <aw_enclave_aws> port 443 user 299" in rules
    assert "pass out quick proto tcp from any to 2001:db8::5 port 8002 user 299" in rules
    assert rules[-1] == "block return out quick proto { tcp, udp } from any to any user 299"
    assert all(rule.startswith("pass") for rule in rules[:-1])


def test_pf_apply_stops_skipping_loopback(firewall, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], str | None]] = []

    class _Done:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    def fake_run(args, input=None, **kwargs):
        calls.append((args, input))
        if args[:3] == ["pfctl", "-s", "Interfaces"]:
            return _Done("lo0 (skip)\nstf0\n")
        if args[:3] == ["pfctl", "-s", "info"]:
            return _Done("Status: Enabled for 0 days\n")
        return _Done()

    monkeypatch.setattr(firewall.subprocess, "run", fake_run)
    firewall._pf_apply("block return out quick proto { tcp, udp } from any to any user 299\n")
    assert calls[0][0] == ["pfctl", "-q", "-a", firewall.PF_ANCHOR, "-f", "-"]
    assert (["pfctl", "-q", "-O", "-f", "-"], "set skip on stf0\n") in calls
    assert not any(args == ["pfctl", "-q", "-e"] for args, _ in calls)
