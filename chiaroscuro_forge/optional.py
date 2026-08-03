"""
Graceful dependency isolation for optional features.

Provides a uniform pattern for importing and using optional third-party
packages. When a dependency is missing, the caller receives a clear
diagnostic message and the feature degrades cleanly rather than crashing
at import time.

Usage
-----
>>> from chiaroscuro_forge.optional import optional_import, requires_optional
>>>
>>> cupy = optional_import("cupy", "CUDA GPU acceleration", "pip install cupy")
>>> if cupy is not None:
...     arr = cupy.asarray([1, 2, 3])
>>>
>>> @requires_optional("fastapi", feature="REST API")
... def serve():
...     import fastapi
...     ...
"""

import functools
import importlib
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def optional_import(
    package: str,
    feature: str = "",
    install_hint: str = "",
    log_level: int = logging.DEBUG,
) -> Optional[Any]:
    """Import *package* if available; return ``None`` otherwise.

    Parameters
    ----------
    package :
        Import path of the package (e.g. ``"cupy"`` or
        ``"fastapi"``).
    feature :
        Human-readable name of the feature that depends on this
        package. Used in log messages.
    install_hint :
        Shell command to install the package (e.g.
        ``"pip install cupy"``). Used in log messages.
    log_level :
        Logging level for the "not available" message. Defaults to
        ``DEBUG`` so that production deployments are not spammed.

    Returns
    -------
    module or None
        The imported module, or ``None`` if the import failed.
    """
    try:
        module = importlib.import_module(package)
        return module
    except (ImportError, ModuleNotFoundError):
        msg = f"{package} is not installed"
        if feature:
            msg = f"{msg}. {feature} is unavailable"
        if install_hint:
            msg = f"{msg}. Install with: {install_hint}"
        logger.log(log_level, msg)
        return None


def is_available(*dependencies: str) -> bool:
    """Return ``True`` if every listed package is importable.

    Parameters
    ----------
    dependencies :
        One or more package names to check.

    Returns
    -------
    bool
    """
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except (ImportError, ModuleNotFoundError):
            return False
    return True


def requires_optional(
    *packages: str,
    feature: str = "",
    install_hint: str = "",
) -> Callable:
    """Decorator: skip the decorated function when dependencies are absent.

    If any required package is missing, calling the function raises
    ``ImportError`` with a message that lists the missing packages.
    Parameters
    ----------
    packages :
        One or more package names required to run the decorated
        function.
    feature :
        Human-readable feature name.
    install_hint :
        Shell command for installing the missing package.

    Returns
    -------
    Callable
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            missing = [pkg for pkg in packages if not is_available(pkg)]

            if missing:
                msg = f"{func.__name__} requires missing packages: " f"{', '.join(missing)}"
                if feature:
                    msg = f"{msg}. {feature} is unavailable"
                if install_hint:
                    msg = f"{msg}. Install with: {install_hint}"
                raise ImportError(msg)

            return func(*args, **kwargs)

        return wrapper

    return decorator
