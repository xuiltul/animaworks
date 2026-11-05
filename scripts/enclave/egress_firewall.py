#!/usr/bin/env python3
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Restrict outbound traffic of the enclave OS user with nftables (Linux) or pf (macOS).

The enclave user may only reach:

* loopback: DNS (53) and the listed ports (its own server, the SSM tunnel port);
* the HOST:PORT pairs given with ``--allow-endpoints`` (e.g. in-house LLM
  servers), TCP only;
* the NAME:PORT pairs given with ``--allow-hosts`` (e.g. a cloud LLM API),
  TCP only.  Names are resolved when the rules are applied, so re-run often
  (e.g. hourly) for endpoints whose addresses move;
* AWS addresses of one region (the published ``AMAZON`` ranges), TCP 443
  only, when ``--region`` is given.  Service endpoints share these ranges
  with EC2, so this cannot tell an AWS API from a host someone runs on EC2 in
  that region; it closes the LAN, the host's other local services and the
  rest of the internet.  Leave ``--region`` out when the enclave needs no AWS.

Only packets owned by that uid enter the restricting chain; packets without a
socket (FIN/ACK of closed sockets of other users) pass untouched.  Everything
else from that uid is rejected, including the host's other local
services (e.g. the main AnimaWorks API) and the LAN.  Run as root::

    python3 egress_firewall.py --user aw-enclave --region ap-northeast-1 \
        --loopback-ports 18610,33061-33070 --apply

Without ``--apply`` the ruleset is printed.  ``--remove`` deletes the table.
Re-run periodically (e.g. weekly) to refresh the AWS ranges.

Several enclaves on one host each need their own rules: pass ``--name`` (the
instance name) so the rules go into a table / anchor of their own.  Without
it the single-instance names below are used, and applying the rules of a
second enclave would replace those of the first.

On macOS pass ``--backend pf``.  The rules are loaded into the sub-anchor
``com.apple/250.AnimaworksEnclave`` (``com.apple/250.AnimaworksEnclave-<name>``
with ``--name``), which the stock ``/etc/pf.conf``
evaluates through its ``anchor "com.apple/*"`` line, so ``pf.conf`` is left
untouched; ``--apply`` also enables pf if it is off and stops pf from
skipping lo0 (otherwise loopback would bypass the rules).  pf matches ``user``
only for TCP and UDP, so the same limits apply to those protocols.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import pwd
import re
import socket
import subprocess
import sys
import urllib.request

AWS_RANGES_URL = "https://ip-ranges.amazonaws.com/ip-ranges.json"
TABLE = "aw_enclave_egress"
PF_ANCHOR = "com.apple/250.AnimaworksEnclave"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def instance_targets(name: str | None) -> tuple[str, str]:
    """Return the ``(nft table, pf anchor)`` that hold one enclave's rules.

    ``None`` keeps the single-instance names so an existing installation is
    left where it is.
    """
    if not name:
        return TABLE, PF_ANCHOR
    if not _NAME_RE.match(name):
        raise ValueError("instance name must match [a-z0-9][a-z0-9_-]{0,31}")
    return f"{TABLE}_{name.replace('-', '_')}", f"{PF_ANCHOR}-{name}"


def _aws_ranges(region: str) -> tuple[list[str], list[str]]:
    with urllib.request.urlopen(AWS_RANGES_URL, timeout=30) as resp:  # noqa: S310 - fixed https URL
        data = json.load(resp)

    def _collect(items: list[dict], key: str) -> list[str]:
        nets = [ipaddress.ip_network(i[key]) for i in items if i["region"] == region and i["service"] == "AMAZON"]
        return [str(n) for n in ipaddress.collapse_addresses(nets)] if nets else []

    v4 = _collect(data["prefixes"], "ip_prefix")
    v6 = _collect(data["ipv6_prefixes"], "ipv6_prefix")
    return v4, v6


def _parse_ports(value: str) -> list[int]:
    """Parse ``18610,33061-33070`` into a list of ports."""
    ports: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        low, sep, high = item.partition("-")
        if sep:
            ports.extend(range(int(low), int(high) + 1))
        else:
            ports.append(int(item))
    return ports


def _region_ranges(region: str | None) -> tuple[list[str], list[str]]:
    """Return the AWS ranges of *region*, or no ranges when no region is given."""
    if not region:
        return [], []
    v4, v6 = _aws_ranges(region)
    if not v4:
        raise SystemExit(f"no AWS ranges found for region {region}")
    return v4, v6


def build_ruleset(
    uid: int,
    region: str | None,
    loopback_ports: list[int],
    endpoints: list[tuple[str, int]] | None = None,
    *,
    table: str = TABLE,
) -> str:
    v4, v6 = _region_ranges(region)
    ports = ", ".join(str(p) for p in sorted({53, *loopback_ports}))
    lines = [f"table inet {table} {{"]
    if v4:
        lines.append("  set aws_v4 { type ipv4_addr; flags interval; elements = { " + ", ".join(v4) + " } }")
    if v6:
        lines.append("  set aws_v6 { type ipv6_addr; flags interval; elements = { " + ", ".join(v6) + " } }")
    lines += [
        "  chain output {",
        "    type filter hook output priority filter; policy accept;",
        # Packets without an owning socket (FIN/ACK from closed sockets,
        # kernel-generated ICMPv6) never match skuid, so only the enclave
        # user's own traffic enters the restricting chain.
        f"    meta skuid {uid} jump enclave",
        "  }",
        "  chain enclave {",
        "    ct state established,related accept",
        f"    oif lo meta l4proto {{ tcp, udp }} th dport {{ {ports} }} accept",
        f"    oif lo meta l4proto {{ tcp, udp }} th sport {{ {ports} }} accept",
    ]
    if v4:
        lines.append("    ip daddr @aws_v4 tcp dport 443 accept")
    for host, port in endpoints or []:
        family = "ip6" if ipaddress.ip_address(host).version == 6 else "ip"
        lines.append(f"    {family} daddr {host} tcp dport {port} accept")
    if v6:
        lines.append("    ip6 daddr @aws_v6 tcp dport 443 accept")
    lines += [
        '    limit rate 6/minute burst 5 packets log prefix "aw-enclave-egress-drop "',
        "    reject",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def build_pf_ruleset(
    uid: int,
    region: str | None,
    loopback_ports: list[int],
    endpoints: list[tuple[str, int]] | None = None,
) -> str:
    v4, v6 = _region_ranges(region)
    ports = ", ".join(str(p) for p in sorted({53, *loopback_ports}))
    lines = []
    if v4:
        # Tables are local to the anchor, so each enclave's anchor has its own.
        lines.append("table <aw_enclave_aws> persist { " + ", ".join([*v4, *v6]) + " }")
    lines += [
        # Only sockets owned by the enclave uid match ``user``, so other
        # users' traffic never reaches the final block rule.
        # Loopback packets leave and re-enter on lo0, so a state created on
        # the way out stalls the handshake (SYN_SENT:CLOSED); match each
        # packet instead.
        f"pass out quick on lo0 proto {{ tcp, udp }} from any to any port {{ {ports} }} user {uid} no state",
        f"pass out quick on lo0 proto {{ tcp, udp }} from any port {{ {ports} }} to any user {uid} no state",
    ]
    if v4:
        lines.append(f"pass out quick proto tcp from any to <aw_enclave_aws> port 443 user {uid}")
    for host, port in endpoints or []:
        lines.append(f"pass out quick proto tcp from any to {host} port {port} user {uid}")
    lines.append(f"block return out quick proto {{ tcp, udp }} from any to any user {uid}")
    return "\n".join(lines) + "\n"


def _pf_apply(ruleset: str, anchor: str = PF_ANCHOR) -> None:
    subprocess.run(["pfctl", "-q", "-a", anchor, "-f", "-"], input=ruleset, text=True, check=True)
    # macOS can mark lo0 as skipped, which exempts loopback from every rule
    # and lets the enclave reach the host's local services.  Loading only the
    # options with an unused interface clears that flag; re-checked on every
    # refresh in case something sets it again.
    interfaces = subprocess.run(["pfctl", "-s", "Interfaces", "-v"], capture_output=True, text=True, check=False)
    if "lo0 (skip)" in interfaces.stdout:
        subprocess.run(["pfctl", "-q", "-O", "-f", "-"], input="set skip on stf0\n", text=True, check=True)
    info = subprocess.run(["pfctl", "-s", "info"], capture_output=True, text=True, check=False)
    if "Status: Enabled" not in info.stdout:
        subprocess.run(["pfctl", "-q", "-e"], check=True)


def _nft(args: list[str], stdin: str | None = None) -> None:
    subprocess.run(["nft", *args], input=stdin, text=True, check=True)


def _resolve_hosts(raw: str) -> list[tuple[str, int]]:
    """Resolve NAME:PORT pairs to every current address (unresolvable names are skipped)."""
    endpoints: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, port = item.rpartition(":")
        try:
            infos = socket.getaddrinfo(name, int(port), proto=socket.IPPROTO_TCP)
        except OSError as exc:
            sys.stderr.write(f"warning: could not resolve {name}: {exc}\n")
            continue
        for addr in sorted({info[4][0] for info in infos}):
            endpoints.append((addr, int(port)))
    return endpoints


def _parse_endpoints(raw: str) -> list[tuple[str, int]]:
    endpoints: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        host = host.strip("[]")
        ipaddress.ip_address(host)  # IP literals only; a name would be resolved once and go stale
        endpoints.append((host, int(port)))
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--name",
        default="",
        help="enclave instance name; gives this enclave its own nft table / pf anchor (needed for a second enclave)",
    )
    parser.add_argument("--region", default="", help="AWS region whose ranges may be reached on 443 (omit for none)")
    parser.add_argument(
        "--loopback-ports", default="", help="comma-separated loopback TCP ports or LOW-HIGH ranges to allow"
    )
    parser.add_argument(
        "--allow-endpoints",
        default="",
        help="comma-separated HOST:PORT (IP literals) the enclave may also reach, e.g. in-house LLM servers",
    )
    parser.add_argument(
        "--allow-hosts",
        default="",
        help="comma-separated NAME:PORT resolved at apply time, e.g. a cloud LLM API",
    )
    parser.add_argument("--backend", choices=("nft", "pf"), default="nft")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    try:
        table, anchor = instance_targets(args.name or None)
    except ValueError as exc:
        parser.error(str(exc))

    if args.remove:
        if args.backend == "pf":
            subprocess.run(["pfctl", "-q", "-a", anchor, "-F", "all"], check=False)
        else:
            subprocess.run(["nft", "delete", "table", "inet", table], check=False)
        return 0

    uid = pwd.getpwnam(args.user).pw_uid
    ports = _parse_ports(args.loopback_ports)
    endpoints = _parse_endpoints(args.allow_endpoints) + _resolve_hosts(args.allow_hosts)
    region = args.region or None
    if args.backend == "pf":
        ruleset = build_pf_ruleset(uid, region, ports, endpoints)
    else:
        ruleset = build_ruleset(uid, region, ports, endpoints, table=table)
    if not args.apply:
        sys.stdout.write(ruleset)
        return 0
    if args.backend == "pf":
        _pf_apply(ruleset, anchor)
        return 0
    # Replace atomically: declare, flush, then load in one transaction.
    _nft(["-f", "-"], f"table inet {table}\ndelete table inet {table}\n{ruleset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
