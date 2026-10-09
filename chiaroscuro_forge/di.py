"""
Dependency Injection Module.

Provides a simple, educational dependency injection container for the
chiaroscuro-forge package. Demonstrates SOLID principles and enables
better testability through inversion of control.

This module is designed for academic teaching purposes, showing how DI
can improve software architecture without over-engineering.

Example
-------
>>> from chiaroscuro_forge.cache import get_cache_manager
>>> from chiaroscuro_forge.di import ServiceContainer, inject
>>>
>>> # Create a container
>>> container = ServiceContainer()
>>>
>>> # Register services
>>> container.register('cache', get_cache_manager())
>>>
>>> # Use injection decorator
>>> @inject('cache')
>>> def my_function(data, cache=None):
...     # cache is automatically injected
...     return cache.get(data)
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

T = TypeVar("T")


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered in the container."""

    pass


class ServiceContainer:
    """
    Simple dependency injection container.

    Manages service registration and retrieval using the Service Locator
    pattern. Designed for educational purposes to demonstrate:
    - Inversion of Control (IoC)
    - Dependency Injection
    - Service Locator pattern

    Attributes
    ----------
    _services : dict
        Mapping of service names to instances
    _factories : dict
        Mapping of service names to factory functions
    _singletons : set
        Service names whose factory-created instances are cached
    """

    def __init__(self):
        """Initialize an empty service container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: set = set()

    def register(self, name: str, service: Any, singleton: bool = True) -> None:
        """
        Register a service instance.

        Parameters
        ----------
        name : str
            Service identifier
        service : Any
            Service instance or value
        singleton : bool, default=True
            Record the name in the singleton set. The instance is stored
            as-is either way, so this flag has no effect on retrieval;
            it only matters for names resolved through a factory

        Examples
        --------
        >>> container = ServiceContainer()
        >>> container.register('config', {'debug': True})
        >>> config = container.get('config')
        """
        self._services[name] = service
        if singleton:
            self._singletons.add(name)

    def register_factory(self, name: str, factory: Callable, singleton: bool = False) -> None:
        """
        Store a factory function for lazy service creation.

        Parameters
        ----------
        name : str
            Service identifier
        factory : Callable
            Callable that creates the service
        singleton : bool, default=False
            Cache the created instance on first use

        Examples
        --------
        >>> def create_cache():
        ...     return CacheManager()
        >>> container.register_factory('cache', create_cache, singleton=True)
        """
        self._factories[name] = factory
        if singleton:
            self._singletons.add(name)

    def get(self, name: str) -> Any:
        """
        Retrieve a service by name.

        Parameters
        ----------
        name : str
            Service identifier

        Returns
        -------
        Any
            The requested service instance

        Raises
        ------
        ServiceNotFoundError
            If service is not registered

        Examples
        --------
        >>> cache = container.get('cache')
        """
        # Return a directly registered or previously cached instance
        if name in self._services:
            return self._services[name]

        # Check if factory exists
        if name in self._factories:
            instance = self._factories[name]()

            if name in self._singletons:
                self._services[name] = instance

            return instance

        raise ServiceNotFoundError(
            f"Service '{name}' not found in container. "
            f"Available services: {list(self._services.keys())}"
        )

    def has(self, name: str) -> bool:
        """
        Check if a service is registered.

        Parameters
        ----------
        name : str
            Service identifier

        Returns
        -------
        bool
            True if service exists, False otherwise
        """
        return name in self._services or name in self._factories

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()

    def list_services(self) -> list:
        """
        Get list of all registered service names.

        Returns
        -------
        list
            Service identifiers
        """
        return list(set(self._services.keys()) | set(self._factories.keys()))


# Global container instance (Service Locator pattern)
_global_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    Get or create the global service container.

    Uses lazy initialization to create container on first access.
    Implements the Singleton pattern for the global container.

    Returns
    -------
    ServiceContainer
        Global container instance

    Examples
    --------
    >>> container = get_container()
    >>> container.register('my_service', MyService())
    """
    global _global_container
    if _global_container is None:
        _global_container = ServiceContainer()
    return _global_container


def reset_container() -> None:
    """
    Reset the global container.

    Useful for testing to ensure clean state between tests.

    Examples
    --------
    >>> reset_container()  # Start fresh
    >>> container = get_container()
    """
    global _global_container
    _global_container = None


def inject(*service_names: str, container: Optional[ServiceContainer] = None):
    """
    Inject services into the decorated function as keyword arguments.

    Demonstrates the Decorator pattern and dependency injection.

    Parameters
    ----------
    *service_names : str
        Names of services to inject
    container : ServiceContainer, optional
        Container to use (defaults to global)

    Returns
    -------
    Callable
        Decorated function with injected dependencies

    Raises
    ------
    ServiceNotFoundError
        If a requested service is not found in the container

    Examples
    --------
    >>> @inject('cache', 'config')
    >>> def process_data(data, cache=None, config=None):
    ...     # cache and config are automatically injected
    ...     if config['use_cache']:
    ...         return cache.get(data)
    ...     return data

    Notes
    -----
    - Services are injected only if the parameter default is None
    - Maintains backward compatibility with explicit arguments
    """

    def decorator(func: Callable) -> Callable:
        # Get function signature
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided container or global
            svc_container = container or get_container()

            # Inject services for parameters with None default
            for service_name in service_names:
                # Check if parameter exists and is not already provided
                if service_name in sig.parameters:
                    param = sig.parameters[service_name]

                    # Only inject if parameter has None default and not provided
                    if param.default is None and service_name not in kwargs:
                        try:
                            kwargs[service_name] = svc_container.get(service_name)
                        except ServiceNotFoundError:
                            # Re-raise with more context
                            raise ServiceNotFoundError(
                                f"Cannot inject '{service_name}' into {func.__name__}: "
                                f"service not found in container"
                            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Service Protocol definitions for type hints
class CacheProtocol(Protocol):
    """Protocol defining cache service interface."""

    def get(self, key: str) -> Any:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


class MetricsProtocol(Protocol):
    """Protocol defining metrics calculation service interface."""

    def calculate(self, original: Any, processed: Any) -> Dict[str, float]:
        """Calculate quality metrics."""
        ...


class ValidationProtocol(Protocol):
    """Protocol defining validation service interface."""

    def validate_image(self, image_path: str) -> bool:
        """Validate image file."""
        ...

    def validate_params(self, **params) -> bool:
        """Validate processing parameters."""
        ...


def setup_default_services() -> ServiceContainer:
    """
    Set up container with default services for the package.

    This function demonstrates how to bootstrap a DI container
    with all necessary services for the application.

    Returns
    -------
    ServiceContainer
        Container configured with default services

    Examples
    --------
    >>> container = setup_default_services()
    >>> # All services are now available
    """
    container = ServiceContainer()

    # Register cache service (lazy initialization)
    def create_cache():
        from .cache import get_cache_manager

        return get_cache_manager()

    container.register_factory("cache", create_cache, singleton=True)

    # Register metrics service
    def create_metrics():
        from . import metrics as metrics_module

        return metrics_module

    container.register_factory("metrics", create_metrics, singleton=True)

    # Register validation service
    def create_validation():
        from . import validation as validation_module

        return validation_module

    container.register_factory("validation", create_validation, singleton=True)

    return container


# Educational examples and patterns
class ExampleService:
    """
    Teaching example of a service designed for dependency injection.

    This class is illustrative only; the package does not use it. It shows
    a service that:
    - Has clear dependencies declared in __init__
    - Can be easily tested with mock dependencies
    - Follows single responsibility principle
    """

    def __init__(self, cache=None, config: Optional[Dict] = None):
        """
        Initialize service with injected dependencies.

        Parameters
        ----------
        cache : optional
            Cache service, if caching is wanted
        config : dict, optional
            Configuration dictionary
        """
        self.cache = cache
        self.config = config or {}

    def process(self, data: Any) -> Any:
        """Process data using injected dependencies."""
        if self.cache and self.config.get("use_cache", False):
            cached = self.cache.get(str(data))
            if cached is not None:
                return cached

        # Process data
        result = self._do_processing(data)

        if self.cache and self.config.get("use_cache", False):
            self.cache.set(str(data), result)

        return result

    def _do_processing(self, data: Any) -> Any:
        """Apply the internal processing logic to the data."""
        return data  # Placeholder
