"""Tests for REST API Python version requirements."""

import sys
from unittest.mock import patch

import pytest

from chiaroscuro_forge.api import run_server


def test_run_server_requires_python_310():
    """Reject Python 3.9 directly and simulate it on newer interpreters."""
    if sys.version_info < (3, 10):
        with pytest.raises(ImportError, match="Python 3.10"):
            run_server()
        return

    with patch("chiaroscuro_forge.api.sys.version_info", (3, 9)):
        with pytest.raises(ImportError, match="Python 3.10"):
            run_server()
