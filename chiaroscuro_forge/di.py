"""
Dependency Injection Module

Provides a simple, educational dependency injection container for the
chiaroscuro-forge package. Demonstrates SOLID principles and enables
better testability through inversion of control.

This module is designed for academic teaching purposes, showing how DI
can improve software architecture without over-engineering.

Example:
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

from typing import Any, Dict, Callable, Optional, TypeVar, Protocol
from functools import wraps
import inspect


T = TypeVar('T')


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
    
    Attributes:
        _services: Dictionary mapping service names to instances
        _factories: Dictionary mapping service names to factory functions
        _singletons: Set of service names that should be singletons
    """
    
    def __init__(self):
        """Initialize an empty service container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: set = set()
    
    def register(
        self,
        name: str,
        service: Any,
        singleton: bool = True
    ) -> None:
        """
        Register a service instance.
        
        Args:
            name: Service identifier
            service: Service instance or value
            singleton: Whether to treat as singleton (default: True)
            
        Example:
            >>> container = ServiceContainer()
            >>> container.register('config', {'debug': True})
            >>> config = container.get('config')
        """
        self._services[name] = service
        if singleton:
            self._singletons.add(name)
    
    def register_factory(
        self,
        name: str,
        factory: Callable,
        singleton: bool = False
    ) -> None:
        """
        Register a factory function for lazy service creation.
        
        Args:
            name: Service identifier
            factory: Callable that creates the service
            singleton: Whether to cache the created instance
            
        Example:
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
        
        Args:
            name: Service identifier
            
        Returns:
            The requested service instance
            
        Raises:
            ServiceNotFoundError: If service is not registered
            
        Example:
            >>> cache = container.get('cache')
        """
        # Check if already instantiated
        if name in self._services:
            return self._services[name]
        
        # Check if factory exists
        if name in self._factories:
            instance = self._factories[name]()
            
            # Cache if singleton
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
        
        Args:
            name: Service identifier
            
        Returns:
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
        
        Returns:
            List of service identifiers
        """
        return list(set(self._services.keys()) | set(self._factories.keys()))


# Global container instance (Service Locator pattern)
_global_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    Get or create the global service container.
    
    Uses lazy initialization to create container on first access.
    Implements the Singleton pattern for the global container.
    
    Returns:
        Global ServiceContainer instance
        
    Example:
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
    
    Example:
        >>> reset_container()  # Start fresh
        >>> container = get_container()
    """
    global _global_container
    _global_container = None


def inject(*service_names: str, container: Optional[ServiceContainer] = None):
    """
    Decorator for automatic dependency injection.
    
    Injects services as keyword arguments to the decorated function.
    Demonstrates the Decorator pattern and dependency injection.
    
    Args:
        *service_names: Names of services to inject
        container: Optional container to use (defaults to global)
        
    Returns:
        Decorated function with injected dependencies
        
    Example:
        >>> @inject('cache', 'config')
        >>> def process_data(data, cache=None, config=None):
        ...     # cache and config are automatically injected
        ...     if config['use_cache']:
        ...         return cache.get(data)
        ...     return data
        
    Note:
        - Services are injected only if the parameter default is None
        - Maintains backward compatibility with explicit arguments
        - Raises ServiceNotFoundError if service not found
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
    Setup container with default services for the package.
    
    This function demonstrates how to bootstrap a DI container
    with all necessary services for the application.
    
    Returns:
        Configured ServiceContainer with default services
        
    Example:
        >>> container = setup_default_services()
        >>> # All services are now available
    """
    container = ServiceContainer()
    
    # Register cache service (lazy initialization)
    def create_cache():
        from .cache import get_cache_manager
        return get_cache_manager()
    
    container.register_factory('cache', create_cache, singleton=True)
    
    # Register metrics service
    def create_metrics():
        from . import metrics as metrics_module
        return metrics_module
    
    container.register_factory('metrics', create_metrics, singleton=True)
    
    # Register validation service
    def create_validation():
        from . import validation as validation_module
        return validation_module
    
    container.register_factory('validation', create_validation, singleton=True)
    
    return container


# Educational examples and patterns
class ExampleService:
    """
    Example service demonstrating DI principles.
    
    This class shows how to design a service that:
    - Has clear dependencies declared in __init__
    - Can be easily tested with mock dependencies
    - Follows single responsibility principle
    """
    
    def __init__(self, cache=None, config: Optional[Dict] = None):
        """
        Initialize service with injected dependencies.
        
        Args:
            cache: Optional cache service
            config: Optional configuration dictionary
        """
        self.cache = cache
        self.config = config or {}
    
    def process(self, data: Any) -> Any:
        """
        Process data using injected dependencies.
        
        Demonstrates how injected dependencies make testing easier.
        """
        if self.cache and self.config.get('use_cache', False):
            cached = self.cache.get(str(data))
            if cached is not None:
                return cached
        
        # Process data
        result = self._do_processing(data)
        
        if self.cache and self.config.get('use_cache', False):
            self.cache.set(str(data), result)
        
        return result
    
    def _do_processing(self, data: Any) -> Any:
        """Internal processing logic."""
        return data  # Placeholder
