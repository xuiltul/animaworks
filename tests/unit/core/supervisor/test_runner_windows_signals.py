from __future__ import annotations

from unittest.mock import patch

from core.supervisor.runner import _install_signal_diagnostics


def test_install_signal_diagnostics_skips_missing_sighup() -> None:
    with patch("signal.signal") as mock_signal:
        _install_signal_diagnostics("sakura")

    registered = [call.args[0] for call in mock_signal.call_args_list]
    assert len(registered) >= 2
