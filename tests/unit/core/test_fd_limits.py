"""Unit tests for core/fd_limits.py — file descriptor limit utilities."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock


from core import fd_limits


class TestEnvInt:
    """Tests for _env_int helper function."""

    def test_returns_default_when_env_var_not_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            result = fd_limits._env_int("NONEXISTENT_VAR", 42)
            assert result == 42

    def test_returns_default_when_env_var_empty(self):
        with mock.patch.dict(os.environ, {"TEST_VAR": ""}, clear=False):
            result = fd_limits._env_int("TEST_VAR", 100)
            assert result == 100

    def test_returns_default_when_env_var_whitespace(self):
        with mock.patch.dict(os.environ, {"TEST_VAR": "   "}, clear=False):
            result = fd_limits._env_int("TEST_VAR", 200)
            assert result == 200

    def test_parses_valid_integer(self):
        with mock.patch.dict(os.environ, {"TEST_VAR": "1234"}, clear=False):
            result = fd_limits._env_int("TEST_VAR", 0)
            assert result == 1234

    def test_returns_default_on_invalid_integer(self, caplog):
        with mock.patch.dict(os.environ, {"TEST_VAR": "not_a_number"}, clear=False):
            result = fd_limits._env_int("TEST_VAR", 999)
            assert result == 999
            assert "Invalid TEST_VAR" in caplog.text


class TestNormalizeLimit:
    """Tests for _normalize_limit helper function."""

    def test_returns_none_for_none_input(self):
        assert fd_limits._normalize_limit(None) is None

    def test_returns_none_for_negative_values(self):
        assert fd_limits._normalize_limit(-1) is None
        assert fd_limits._normalize_limit(-999) is None

    def test_returns_int_for_positive_values(self):
        assert fd_limits._normalize_limit(1024) == 1024
        assert fd_limits._normalize_limit(0) == 0

    @mock.patch("core.fd_limits.resource")
    def test_returns_none_for_rlim_infinity(self, mock_resource):
        mock_resource.RLIM_INFINITY = 2**63 - 1
        assert fd_limits._normalize_limit(2**63 - 1) is None

    def test_handles_resource_none(self):
        with mock.patch("core.fd_limits.resource", None):
            assert fd_limits._normalize_limit(1024) == 1024


class TestGetNofileLimits:
    """Tests for get_nofile_limits function."""

    def test_returns_none_tuple_when_resource_unavailable(self):
        with mock.patch("core.fd_limits.resource", None):
            result = fd_limits.get_nofile_limits()
            assert result == (None, None)

    @mock.patch("core.fd_limits.resource")
    def test_returns_soft_hard_limits(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.getrlimit.return_value = (1024, 4096)
        mock_resource.RLIM_INFINITY = 2**63 - 1

        result = fd_limits.get_nofile_limits()
        assert result == (1024, 4096)
        mock_resource.getrlimit.assert_called_once_with(7)

    @mock.patch("core.fd_limits.resource")
    def test_normalizes_infinity_to_none(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (2**63 - 1, 2**63 - 1)

        result = fd_limits.get_nofile_limits()
        assert result == (None, None)

    @mock.patch("core.fd_limits.resource")
    def test_returns_none_tuple_on_exception(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.getrlimit.side_effect = OSError("Permission denied")

        result = fd_limits.get_nofile_limits()
        assert result == (None, None)


class TestRaiseNofileSoftLimit:
    """Tests for raise_nofile_soft_limit function."""

    def test_returns_none_tuple_when_resource_unavailable(self):
        with mock.patch("core.fd_limits.resource", None):
            result = fd_limits.raise_nofile_soft_limit()
            assert result == (None, None, None)

    @mock.patch("core.fd_limits.resource")
    def test_uses_min_soft_parameter(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (256, 4096)

        result = fd_limits.raise_nofile_soft_limit(min_soft=2048)

        assert result == (256, 2048, 4096)
        mock_resource.setrlimit.assert_called_once_with(7, (2048, 4096))

    @mock.patch("core.fd_limits.resource")
    def test_uses_env_var_when_min_soft_not_provided(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (256, 16384)

        with mock.patch.dict(os.environ, {"ANIMAWORKS_NOFILE_SOFT": "4096"}, clear=False):
            result = fd_limits.raise_nofile_soft_limit()

        assert result == (256, 4096, 16384)
        mock_resource.setrlimit.assert_called_once_with(7, (4096, 16384))

    @mock.patch("core.fd_limits.resource")
    def test_defaults_to_8192_when_no_env_var(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (256, 16384)

        with mock.patch.dict(os.environ, {}, clear=False):
            if "ANIMAWORKS_NOFILE_SOFT" in os.environ:
                del os.environ["ANIMAWORKS_NOFILE_SOFT"]
            result = fd_limits.raise_nofile_soft_limit()

        assert result == (256, 8192, 16384)

    @mock.patch("core.fd_limits.resource")
    def test_does_not_raise_when_already_unlimited(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (2**63 - 1, 2**63 - 1)

        result = fd_limits.raise_nofile_soft_limit(min_soft=2048)

        assert result == (None, None, None)
        mock_resource.setrlimit.assert_not_called()

    @mock.patch("core.fd_limits.resource")
    def test_does_not_raise_when_already_sufficient(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (4096, 8192)

        result = fd_limits.raise_nofile_soft_limit(min_soft=2048)

        assert result == (4096, 4096, 8192)
        mock_resource.setrlimit.assert_not_called()

    @mock.patch("core.fd_limits.resource")
    def test_caps_at_hard_limit(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (256, 1024)

        result = fd_limits.raise_nofile_soft_limit(min_soft=8192)

        # Should cap at hard limit (1024), not requested 8192
        assert result == (256, 1024, 1024)
        mock_resource.setrlimit.assert_called_once_with(7, (1024, 1024))

    @mock.patch("core.fd_limits.resource")
    def test_enforces_minimum_of_64(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (32, 4096)

        result = fd_limits.raise_nofile_soft_limit(min_soft=10)

        # Should use 64 as minimum, not 10
        assert result == (32, 64, 4096)
        mock_resource.setrlimit.assert_called_once_with(7, (64, 4096))

    @mock.patch("core.fd_limits.resource")
    def test_returns_old_limit_on_getrlimit_exception(self, mock_resource):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.getrlimit.side_effect = OSError("Permission denied")

        result = fd_limits.raise_nofile_soft_limit()
        assert result == (None, None, None)

    @mock.patch("core.fd_limits.resource")
    def test_returns_old_limit_on_setrlimit_exception(self, mock_resource, caplog):
        mock_resource.RLIMIT_NOFILE = 7
        mock_resource.RLIM_INFINITY = 2**63 - 1
        mock_resource.getrlimit.return_value = (256, 4096)
        mock_resource.setrlimit.side_effect = OSError("Operation not permitted")

        result = fd_limits.raise_nofile_soft_limit(min_soft=2048)

        assert result == (256, 256, 4096)
        assert "Failed to raise RLIMIT_NOFILE" in caplog.text


class TestCountOpenFds:
    """Tests for count_open_fds function."""

    def test_counts_fds_from_proc_self_fd(self):
        with mock.patch("os.listdir") as mock_listdir, mock.patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True
            mock_listdir.return_value = ["0", "1", "2", "3", "not_a_number", ".", ".."]

            result = fd_limits.count_open_fds()

            assert result == 4  # Only numeric entries

    def test_counts_fds_from_dev_fd_when_proc_unavailable(self):
        with mock.patch("os.listdir") as mock_listdir:
            # Mock Path.exists to return False for /proc/self/fd, True for /dev/fd
            original_path_init = Path.__init__

            def mock_path_exists(self):
                path_str = str(self)
                if "/proc/self/fd" in path_str:
                    return False
                if "/dev/fd" in path_str:
                    return True
                return False

            with mock.patch.object(Path, "exists", mock_path_exists):
                mock_listdir.return_value = ["0", "1", "2"]
                result = fd_limits.count_open_fds()
                assert result == 3

    def test_returns_none_when_no_fd_path_available(self):
        with mock.patch.object(Path, "exists", return_value=False):
            result = fd_limits.count_open_fds()
            assert result is None

    def test_returns_none_on_listdir_exception(self):
        with mock.patch.object(Path, "exists", return_value=True), mock.patch(
            "os.listdir", side_effect=PermissionError("Access denied")
        ):
            result = fd_limits.count_open_fds()
            assert result is None

    def test_handles_mixed_numeric_and_non_numeric_entries(self):
        with mock.patch("os.listdir") as mock_listdir, mock.patch.object(Path, "exists", return_value=True):
            mock_listdir.return_value = ["0", "1", "10", "100", ".", "..", "lock", "status"]

            result = fd_limits.count_open_fds()

            assert result == 4  # 0, 1, 10, 100


class TestFdUsageRatio:
    """Tests for fd_usage_ratio function."""

    def test_returns_ratio_when_both_values_known(self):
        result = fd_limits.fd_usage_ratio(50, 100)
        assert result == 0.5

    def test_returns_none_when_open_fds_none(self):
        result = fd_limits.fd_usage_ratio(None, 100)
        assert result is None

    def test_returns_none_when_soft_limit_none(self):
        result = fd_limits.fd_usage_ratio(50, None)
        assert result is None

    def test_returns_none_when_both_none(self):
        result = fd_limits.fd_usage_ratio(None, None)
        assert result is None

    def test_returns_none_when_soft_limit_zero(self):
        result = fd_limits.fd_usage_ratio(50, 0)
        assert result is None

    def test_returns_none_when_soft_limit_negative(self):
        result = fd_limits.fd_usage_ratio(50, -1)
        assert result is None

    def test_handles_zero_open_fds(self):
        result = fd_limits.fd_usage_ratio(0, 100)
        assert result == 0.0

    def test_handles_ratio_greater_than_one(self):
        result = fd_limits.fd_usage_ratio(150, 100)
        assert result == 1.5


class TestFdHeadroom:
    """Tests for fd_headroom function."""

    def test_returns_headroom_when_both_values_known(self):
        result = fd_limits.fd_headroom(50, 100)
        assert result == 50

    def test_returns_none_when_open_fds_none(self):
        result = fd_limits.fd_headroom(None, 100)
        assert result is None

    def test_returns_none_when_soft_limit_none(self):
        result = fd_limits.fd_headroom(50, None)
        assert result is None

    def test_returns_none_when_both_none(self):
        result = fd_limits.fd_headroom(None, None)
        assert result is None

    def test_returns_zero_when_at_limit(self):
        result = fd_limits.fd_headroom(100, 100)
        assert result == 0

    def test_returns_negative_when_over_limit(self):
        result = fd_limits.fd_headroom(150, 100)
        assert result == -50

    def test_handles_zero_open_fds(self):
        result = fd_limits.fd_headroom(0, 100)
        assert result == 100
