"""Unit tests for core/fd_limits.py"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Mock resource module for non-Unix platforms
import sys
sys.modules['resource'] = MagicMock()

from core import fd_limits


class TestGetNofileLimits(unittest.TestCase):
    """Tests for get_nofile_limits()"""

    @patch('core.fd_limits.resource')
    def test_returns_soft_hard_tuple(self, mock_resource):
        """Should return (soft, hard) tuple from getrlimit"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (1024, 4096)
        
        result = fd_limits.get_nofile_limits()
        
        self.assertEqual(result, (1024, 4096))
        mock_resource.getrlimit.assert_called_once_with(mock_resource.RLIMIT_NOFILE)

    @patch('core.fd_limits.resource', None)
    def test_returns_none_none_when_resource_unavailable(self):
        """Should return (None, None) when resource module is not available"""
        result = fd_limits.get_nofile_limits()
        self.assertEqual(result, (None, None))

    @patch('core.fd_limits.resource')
    def test_returns_none_none_on_exception(self, mock_resource):
        """Should return (None, None) when getrlimit raises exception"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.getrlimit.side_effect = OSError("Failed")
        
        result = fd_limits.get_nofile_limits()
        self.assertEqual(result, (None, None))

    @patch('core.fd_limits.resource')
    def test_normalizes_rlim_infinity(self, mock_resource):
        """Should convert RLIM_INFINITY to None"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (-1, -1)  # RLIM_INFINITY
        
        result = fd_limits.get_nofile_limits()
        self.assertEqual(result, (None, None))

    @patch('core.fd_limits.resource')
    def test_normalizes_negative_values(self, mock_resource):
        """Should convert negative values to None"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (-5, 1024)
        
        result = fd_limits.get_nofile_limits()
        self.assertEqual(result, (None, 1024))


class TestRaiseNofileSoftLimit(unittest.TestCase):
    """Tests for raise_nofile_soft_limit()"""

    @patch('core.fd_limits.resource')
    def test_raises_soft_limit_to_target(self, mock_resource):
        """Should raise soft limit to target value"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (1024, 8192)
        mock_resource.setrlimit.return_value = None
        
        old_soft, new_soft, hard = fd_limits.raise_nofile_soft_limit(min_soft=4096)
        
        self.assertEqual(old_soft, 1024)
        self.assertEqual(new_soft, 4096)
        self.assertEqual(hard, 8192)
        mock_resource.setrlimit.assert_called_once()

    @patch('core.fd_limits.resource')
    def test_does_not_exceed_hard_limit(self, mock_resource):
        """Should not exceed hard limit"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (1024, 2048)
        mock_resource.setrlimit.return_value = None
        
        old_soft, new_soft, hard = fd_limits.raise_nofile_soft_limit(min_soft=8192)
        
        # Should cap at hard limit (2048)
        self.assertEqual(new_soft, 2048)

    @patch('core.fd_limits.resource')
    def test_returns_unchanged_if_already_at_target(self, mock_resource):
        """Should not change if already at or above target"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (8192, 8192)
        
        old_soft, new_soft, hard = fd_limits.raise_nofile_soft_limit(min_soft=4096)
        
        self.assertEqual(old_soft, 8192)
        self.assertEqual(new_soft, 8192)
        mock_resource.setrlimit.assert_not_called()

    @patch('core.fd_limits.resource')
    def test_handles_setrlimit_failure(self, mock_resource):
        """Should handle setrlimit failure gracefully"""
        mock_resource.RLIMIT_NOFILE = 1
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (1024, 8192)
        mock_resource.setrlimit.side_effect = OSError("Permission denied")
        
        old_soft, new_soft, hard = fd_limits.raise_nofile_soft_limit(min_soft=4096)
        
        # Should return unchanged values on failure
        self.assertEqual(old_soft, 1024)
        self.assertEqual(new_soft, 1024)
        self.assertEqual(hard, 8192)

    @patch('core.fd_limits.resource', None)
    def test_returns_none_tuple_when_resource_unavailable(self):
        """Should return (None, None, None) when resource module is not available"""
        result = fd_limits.raise_nofile_soft_limit(min_soft=4096)
        self.assertEqual(result, (None, None, None))


class TestCountOpenFds(unittest.TestCase):
    """Tests for count_open_fds()"""

    @patch('core.fd_limits.Path')
    def test_counts_fds_from_proc_self_fd(self, mock_path_class):
        """Should count numeric entries in /proc/self/fd"""
        mock_fd_path = MagicMock()
        mock_fd_path.exists.return_value = True
        mock_fd_path.__iter__ = lambda self: iter(['0', '1', '2', '3', '10', 'txt'])
        mock_path_class.side_effect = lambda p: mock_fd_path if p == '/proc/self/fd' else MagicMock(exists=False)
        
        with patch('os.listdir', return_value=['0', '1', '2', '3', '10', 'txt']):
            result = fd_limits.count_open_fds()
        
        # Should only count numeric entries
        self.assertEqual(result, 5)

    @patch('core.fd_limits.Path')
    def test_falls_back_to_dev_fd(self, mock_path_class):
        """Should fall back to /dev/fd if /proc/self/fd doesn't exist"""
        def path_factory(p):
            mock = MagicMock()
            if p == '/proc/self/fd':
                mock.exists.return_value = False
            elif p == '/dev/fd':
                mock.exists.return_value = True
            return mock
        
        mock_path_class.side_effect = path_factory
        
        with patch('os.listdir', return_value=['0', '1', '2']):
            result = fd_limits.count_open_fds()
        
        self.assertEqual(result, 3)

    @patch('core.fd_limits.Path')
    def test_returns_none_when_no_fd_paths_available(self, mock_path_class):
        """Should return None when no FD paths are available"""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path
        
        result = fd_limits.count_open_fds()
        self.assertIsNone(result)

    @patch('core.fd_limits.Path')
    def test_handles_listdir_exception(self, mock_path_class):
        """Should handle exceptions gracefully"""
        mock_fd_path = MagicMock()
        mock_fd_path.exists.return_value = True
        mock_path_class.return_value = mock_fd_path
        
        with patch('os.listdir', side_effect=PermissionError("Access denied")):
            result = fd_limits.count_open_fds()
        
        self.assertIsNone(result)


class TestFdUsageRatio(unittest.TestCase):
    """Tests for fd_usage_ratio()"""

    def test_calculates_ratio_correctly(self):
        """Should calculate open_fds / soft_limit"""
        result = fd_limits.fd_usage_ratio(open_fds=512, soft_limit=1024)
        self.assertEqual(result, 0.5)

    def test_returns_none_when_open_fds_is_none(self):
        """Should return None when open_fds is None"""
        result = fd_limits.fd_usage_ratio(open_fds=None, soft_limit=1024)
        self.assertIsNone(result)

    def test_returns_none_when_soft_limit_is_none(self):
        """Should return None when soft_limit is None"""
        result = fd_limits.fd_usage_ratio(open_fds=512, soft_limit=None)
        self.assertIsNone(result)

    def test_returns_none_when_soft_limit_is_zero(self):
        """Should return None when soft_limit is zero or negative"""
        result = fd_limits.fd_usage_ratio(open_fds=512, soft_limit=0)
        self.assertIsNone(result)
        
        result = fd_limits.fd_usage_ratio(open_fds=512, soft_limit=-1)
        self.assertIsNone(result)


class TestFdHeadroom(unittest.TestCase):
    """Tests for fd_headroom()"""

    def test_calculates_headroom_correctly(self):
        """Should calculate soft_limit - open_fds"""
        result = fd_limits.fd_headroom(open_fds=512, soft_limit=1024)
        self.assertEqual(result, 512)

    def test_returns_zero_when_at_limit(self):
        """Should return 0 when at limit"""
        result = fd_limits.fd_headroom(open_fds=1024, soft_limit=1024)
        self.assertEqual(result, 0)

    def test_returns_negative_when_over_limit(self):
        """Should return negative value when over limit"""
        result = fd_limits.fd_headroom(open_fds=1500, soft_limit=1024)
        self.assertEqual(result, -476)

    def test_returns_none_when_open_fds_is_none(self):
        """Should return None when open_fds is None"""
        result = fd_limits.fd_headroom(open_fds=None, soft_limit=1024)
        self.assertIsNone(result)

    def test_returns_none_when_soft_limit_is_none(self):
        """Should return None when soft_limit is None"""
        result = fd_limits.fd_headroom(open_fds=512, soft_limit=None)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
