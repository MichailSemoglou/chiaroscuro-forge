"""
Tests for Dependency Injection Module

Comprehensive tests demonstrating DI patterns and ensuring container functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock
from chiaroscuro_forge.di import (
    ServiceContainer,
    ServiceNotFoundError,
    get_container,
    reset_container,
    inject,
    setup_default_services,
    ExampleService,
)


class TestServiceContainer(unittest.TestCase):
    """Test the ServiceContainer class."""

    def setUp(self):
        """Create a fresh container for each test."""
        self.container = ServiceContainer()

    def test_register_and_get_service(self):
        """Test basic service registration and retrieval."""
        service = {"key": "value"}
        self.container.register("test_service", service)

        retrieved = self.container.get("test_service")
        self.assertEqual(retrieved, service)
        self.assertIs(retrieved, service)  # Same instance

    def test_register_singleton(self):
        """Test singleton behavior."""
        service = Mock()
        self.container.register("singleton", service, singleton=True)

        # Should return same instance
        retrieved1 = self.container.get("singleton")
        retrieved2 = self.container.get("singleton")

        self.assertIs(retrieved1, retrieved2)
        self.assertIs(retrieved1, service)

    def test_register_factory(self):
        """Test factory registration."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        self.container.register_factory("factory_service", factory, singleton=False)

        # Each get should create new instance
        service1 = self.container.get("factory_service")
        service2 = self.container.get("factory_service")

        self.assertNotEqual(service1, service2)
        self.assertEqual(service1["count"], 1)
        self.assertEqual(service2["count"], 2)

    def test_register_factory_singleton(self):
        """Test factory with singleton behavior."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        self.container.register_factory("factory_singleton", factory, singleton=True)

        # Should return same instance
        service1 = self.container.get("factory_singleton")
        service2 = self.container.get("factory_singleton")

        self.assertIs(service1, service2)
        self.assertEqual(call_count, 1)  # Factory called only once

    def test_get_nonexistent_service(self):
        """Test error when getting non-existent service."""
        with self.assertRaises(ServiceNotFoundError) as cm:
            self.container.get("nonexistent")

        self.assertIn("nonexistent", str(cm.exception))

    def test_has_service(self):
        """Test checking if service exists."""
        self.assertFalse(self.container.has("test"))

        self.container.register("test", "value")
        self.assertTrue(self.container.has("test"))

        self.container.register_factory("factory", lambda: "value")
        self.assertTrue(self.container.has("factory"))

    def test_clear(self):
        """Test clearing all services."""
        self.container.register("service1", "value1")
        self.container.register("service2", "value2")

        self.assertEqual(len(self.container.list_services()), 2)

        self.container.clear()

        self.assertEqual(len(self.container.list_services()), 0)
        self.assertFalse(self.container.has("service1"))

    def test_list_services(self):
        """Test listing all registered services."""
        self.assertEqual(self.container.list_services(), [])

        self.container.register("service1", "value1")
        self.container.register_factory("service2", lambda: "value2")

        services = self.container.list_services()
        self.assertEqual(set(services), {"service1", "service2"})


class TestGlobalContainer(unittest.TestCase):
    """Test global container functions."""

    def setUp(self):
        """Reset global container before each test."""
        reset_container()

    def tearDown(self):
        """Clean up after each test."""
        reset_container()

    def test_get_container_singleton(self):
        """Test that get_container returns same instance."""
        container1 = get_container()
        container2 = get_container()

        self.assertIs(container1, container2)

    def test_reset_container(self):
        """Test resetting global container."""
        container1 = get_container()
        container1.register("test", "value")

        reset_container()

        container2 = get_container()
        self.assertIsNot(container1, container2)
        self.assertFalse(container2.has("test"))

    def test_container_persists_across_calls(self):
        """Test that container state persists."""
        container = get_container()
        container.register("persistent", "data")

        # Get container again
        container2 = get_container()
        self.assertEqual(container2.get("persistent"), "data")


class TestInjectDecorator(unittest.TestCase):
    """Test the @inject decorator."""

    def setUp(self):
        """Setup fresh container with test services."""
        reset_container()
        self.container = get_container()
        self.container.register("service1", "injected_value1")
        self.container.register("service2", "injected_value2")

    def tearDown(self):
        """Clean up."""
        reset_container()

    def test_inject_single_service(self):
        """Test injecting a single service."""

        @inject("service1")
        def test_func(data, service1=None):
            return f"{data}:{service1}"

        result = test_func("test")
        self.assertEqual(result, "test:injected_value1")

    def test_inject_multiple_services(self):
        """Test injecting multiple services."""

        @inject("service1", "service2")
        def test_func(service1=None, service2=None):
            return f"{service1}:{service2}"

        result = test_func()
        self.assertEqual(result, "injected_value1:injected_value2")

    def test_inject_with_explicit_arg(self):
        """Test that explicit args override injection."""

        @inject("service1")
        def test_func(service1=None):
            return service1

        result = test_func(service1="explicit")
        self.assertEqual(result, "explicit")

    def test_inject_with_custom_container(self):
        """Test injection with custom container."""
        custom_container = ServiceContainer()
        custom_container.register("custom", "custom_value")

        @inject("custom", container=custom_container)
        def test_func(custom=None):
            return custom

        result = test_func()
        self.assertEqual(result, "custom_value")

    def test_inject_nonexistent_service(self):
        """Test error when injecting non-existent service."""

        @inject("nonexistent")
        def test_func(nonexistent=None):
            return nonexistent

        with self.assertRaises(ServiceNotFoundError) as cm:
            test_func()

        self.assertIn("nonexistent", str(cm.exception))
        self.assertIn("test_func", str(cm.exception))

    def test_inject_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @inject("service1")
        def documented_function(service1=None):
            """This is documentation."""
            return service1

        self.assertEqual(documented_function.__name__, "documented_function")
        self.assertEqual(documented_function.__doc__, "This is documentation.")

    def test_inject_with_args_and_kwargs(self):
        """Test injection with mixed args and kwargs."""

        @inject("service1")
        def test_func(arg1, arg2, service1=None, kwarg1="default"):
            return f"{arg1}:{arg2}:{service1}:{kwarg1}"

        result = test_func("a", "b", kwarg1="c")
        self.assertEqual(result, "a:b:injected_value1:c")

    def test_inject_does_not_inject_non_none_defaults(self):
        """Test that services with non-None defaults are not injected."""

        @inject("service1")
        def test_func(service1="default"):
            return service1

        result = test_func()
        self.assertEqual(result, "default")  # Not injected


class TestSetupDefaultServices(unittest.TestCase):
    """Test default service setup."""

    def test_setup_creates_container(self):
        """Test that setup creates a container."""
        container = setup_default_services()
        self.assertIsInstance(container, ServiceContainer)

    def test_setup_registers_cache(self):
        """Test that cache service is registered."""
        container = setup_default_services()
        self.assertTrue(container.has("cache"))

    def test_setup_registers_metrics(self):
        """Test that metrics service is registered."""
        container = setup_default_services()
        self.assertTrue(container.has("metrics"))

    def test_setup_registers_validation(self):
        """Test that validation service is registered."""
        container = setup_default_services()
        self.assertTrue(container.has("validation"))

    def test_cache_service_is_lazy(self):
        """Test that cache service uses lazy initialization."""
        container = setup_default_services()

        # Service should not be instantiated yet
        self.assertNotIn("cache", container._services)

        # Getting service should instantiate it
        cache = container.get("cache")
        self.assertIsNotNone(cache)

    def test_services_are_singletons(self):
        """Test that default services are singletons."""
        container = setup_default_services()

        cache1 = container.get("cache")
        cache2 = container.get("cache")

        self.assertIs(cache1, cache2)


class TestExampleService(unittest.TestCase):
    """Test the ExampleService class."""

    def test_example_service_creation(self):
        """Test creating ExampleService with dependencies."""
        cache = Mock()
        config = {"use_cache": True}

        service = ExampleService(cache=cache, config=config)

        self.assertIs(service.cache, cache)
        self.assertEqual(service.config, config)

    def test_example_service_without_dependencies(self):
        """Test creating ExampleService without dependencies."""
        service = ExampleService()

        self.assertIsNone(service.cache)
        self.assertEqual(service.config, {})

    def test_example_service_process_with_cache(self):
        """Test ExampleService.process with caching."""
        cache = Mock()
        cache.get.return_value = None
        config = {"use_cache": True}

        service = ExampleService(cache=cache, config=config)
        result = service.process("test_data")

        cache.get.assert_called_once_with("test_data")
        cache.set.assert_called_once()

    def test_example_service_process_without_cache(self):
        """Test ExampleService.process without caching."""
        config = {"use_cache": False}
        service = ExampleService(config=config)

        result = service.process("test_data")
        self.assertEqual(result, "test_data")

    def test_example_service_cache_hit(self):
        """Test ExampleService with cache hit."""
        cache = Mock()
        cache.get.return_value = "cached_result"
        config = {"use_cache": True}

        service = ExampleService(cache=cache, config=config)
        result = service.process("test_data")

        self.assertEqual(result, "cached_result")
        cache.set.assert_not_called()  # Should not set if hit


class TestDIPatterns(unittest.TestCase):
    """Test various DI patterns and use cases."""

    def setUp(self):
        """Setup for pattern tests."""
        reset_container()

    def tearDown(self):
        """Cleanup."""
        reset_container()

    def test_constructor_injection_pattern(self):
        """Test constructor injection pattern (recommended)."""

        class ServiceWithDependency:
            def __init__(self, cache, config=None):
                self.cache = cache
                self.config = config or {}

        cache = Mock()
        service = ServiceWithDependency(cache=cache)

        self.assertIs(service.cache, cache)

    def test_property_injection_pattern(self):
        """Test property injection pattern."""

        class ServiceWithProperty:
            def __init__(self):
                self._cache = None

            @property
            def cache(self):
                if self._cache is None:
                    self._cache = get_container().get("cache")
                return self._cache

        get_container().register("cache", Mock())
        service = ServiceWithProperty()

        cache1 = service.cache
        cache2 = service.cache
        self.assertIs(cache1, cache2)  # Should be same instance

    def test_method_injection_pattern(self):
        """Test method injection pattern."""

        @inject("cache")
        def process_with_cache(data, cache=None):
            cache.process(data)
            return "processed"

        cache = Mock()
        get_container().register("cache", cache)

        result = process_with_cache("test")

        cache.process.assert_called_once_with("test")
        self.assertEqual(result, "processed")

    def test_factory_pattern_with_di(self):
        """Test factory pattern using DI."""

        class ServiceFactory:
            def __init__(self, container):
                self.container = container

            def create_service(self, service_type):
                if service_type == "cache":
                    return self.container.get("cache")
                elif service_type == "metrics":
                    return self.container.get("metrics")
                return None

        container = setup_default_services()
        factory = ServiceFactory(container)

        cache = factory.create_service("cache")
        self.assertIsNotNone(cache)

    def test_circular_dependency_prevention(self):
        """Test that circular dependencies can be managed."""
        container = ServiceContainer()

        # Use factory with lazy initialization to prevent circular deps
        class ServiceA:
            def __init__(self):
                self._b = None

            def get_b(self):
                if self._b is None:
                    self._b = container.get("service_b")
                return self._b

        class ServiceB:
            def __init__(self):
                self._a = None

            def get_a(self):
                if self._a is None:
                    self._a = container.get("service_a")
                return self._a

        container.register("service_a", ServiceA())
        container.register("service_b", ServiceB())

        # Should not cause infinite recursion
        a = container.get("service_a")
        b = a.get_b()
        a2 = b.get_a()

        self.assertIs(a, a2)


if __name__ == "__main__":
    unittest.main()
