"""
Dependency Injection Examples

Practical examples demonstrating how to use dependency injection
in chiaroscuro-forge for better testability and flexibility.
"""

# Example 1: Basic Service Registration and Retrieval
def example_basic_di():
    """Show basic DI container usage."""
    from chiaroscuro_forge.di import ServiceContainer
    
    # Create a container
    container = ServiceContainer()
    
    # Register services
    container.register('config', {'debug': True, 'cache_enabled': True})
    container.register('app_name', 'ChiaroscuroForge')
    
    # Retrieve services
    config = container.get('config')
    app_name = container.get('app_name')
    
    print(f"App: {app_name}, Debug: {config['debug']}")


# Example 2: Using the @inject Decorator
def example_inject_decorator():
    """Show how to use the @inject decorator."""
    from chiaroscuro_forge.di import get_container, inject
    
    # Setup container
    container = get_container()
    container.register('cache', {'data': 'cached_value'})
    container.register('logger', print)  # Simple logger
    
    # Define function with injectable dependencies
    @inject('cache', 'logger')
    def process_data(data, cache=None, logger=None):
        """Process data with injected dependencies."""
        logger(f"Processing: {data}")
        
        # Use cache
        if cache and 'data' in cache:
            logger(f"Cache hit: {cache['data']}")
            return cache['data']
        
        return f"Processed: {data}"
    
    # Call function - dependencies are automatically injected
    result = process_data("test_input")
    print(f"Result: {result}")


# Example 3: Factory Pattern with Lazy Initialization
def example_factory_pattern():
    """Show factory pattern for lazy service creation."""
    from chiaroscuro_forge.di import ServiceContainer
    
    container = ServiceContainer()
    
    # Register factory for expensive service
    def create_expensive_service():
        print("Creating expensive service...")
        return {'initialized': True, 'data': [1, 2, 3]}
    
    # Singleton factory - created once and cached
    container.register_factory('expensive', create_expensive_service, singleton=True)
    
    # Service not created yet
    print("Service registered but not created")
    
    # First access creates the service
    service1 = container.get('expensive')
    print(f"First access: {service1}")
    
    # Second access returns cached instance
    service2 = container.get('expensive')
    print(f"Second access (cached): {service2}")
    print(f"Same instance: {service1 is service2}")


# Example 4: Testing with Dependency Injection
def example_testing_with_di():
    """Show how DI improves testability."""
    from chiaroscuro_forge.di import inject, get_container, reset_container
    
    # Reset for clean test
    reset_container()
    
    # Define a function that depends on external services
    @inject('cache', 'metrics')
    def analyze_image(image_path, cache=None, metrics=None):
        """Analyze image with injectable dependencies."""
        # Check cache first
        cached = cache.get(image_path) if cache else None
        if cached:
            return cached
        
        # Simulate processing
        result = {'quality': 0.95, 'processing_time': 0.5}
        
        # Record metrics
        if metrics:
            metrics.record('images_processed', 1)
        
        # Cache result
        if cache:
            cache.set(image_path, result)
        
        return result
    
    # In production: use real services
    from unittest.mock import Mock
    
    # In tests: inject mocks
    mock_cache = Mock()
    mock_cache.get.return_value = None
    mock_cache.set = Mock()
    
    mock_metrics = Mock()
    
    container = get_container()
    container.register('cache', mock_cache)
    container.register('metrics', mock_metrics)
    
    # Test the function
    result = analyze_image('test.jpg')
    
    print(f"Result: {result}")
    print(f"Cache.get called: {mock_cache.get.called}")
    print(f"Cache.set called: {mock_cache.set.called}")
    print(f"Metrics.record called: {mock_metrics.record.called}")


# Example 5: Constructor Injection (Recommended Pattern)
class ImageProcessor:
    """Example class using constructor injection."""
    
    def __init__(self, cache=None, validator=None, metrics=None):
        """
        Initialize with injected dependencies.
        
        Args:
            cache: Cache service for results
            validator: Validation service for inputs
            metrics: Metrics service for monitoring
        """
        self.cache = cache
        self.validator = validator
        self.metrics = metrics
    
    def process(self, image_path):
        """Process image with dependency injection benefits."""
        # Validate input (if validator available)
        if self.validator and not self.validator.validate_path(image_path):
            raise ValueError(f"Invalid path: {image_path}")
        
        # Check cache (if cache available)
        if self.cache:
            cached = self.cache.get(image_path)
            if cached:
                if self.metrics:
                    self.metrics.record('cache_hits', 1)
                return cached
        
        # Process image
        result = self._do_processing(image_path)
        
        # Store in cache (if cache available)
        if self.cache:
            self.cache.set(image_path, result)
        
        # Record metrics (if metrics available)
        if self.metrics:
            self.metrics.record('images_processed', 1)
        
        return result
    
    def _do_processing(self, image_path):
        """Internal processing logic."""
        return {'processed': True, 'path': image_path}


def example_constructor_injection():
    """Show constructor injection pattern."""
    from unittest.mock import Mock
    
    # Create mock dependencies for testing
    mock_cache = Mock()
    mock_cache.get.return_value = None
    
    mock_validator = Mock()
    mock_validator.validate_path.return_value = True
    
    mock_metrics = Mock()
    
    # Inject dependencies via constructor
    processor = ImageProcessor(
        cache=mock_cache,
        validator=mock_validator,
        metrics=mock_metrics
    )
    
    # Use the processor
    result = processor.process('test.jpg')
    
    print(f"Result: {result}")
    print(f"Validator called: {mock_validator.validate_path.called}")
    print(f"Metrics recorded: {mock_metrics.record.called}")


# Example 6: Using Default Services
def example_default_services():
    """Show how to use the default service setup."""
    from chiaroscuro_forge.di import setup_default_services, inject
    
    # Setup default services for the package
    container = setup_default_services()
    
    # List available services
    print("Available services:", container.list_services())
    
    # Use injected services
    @inject('cache')
    def process_with_cache(data, cache=None):
        cache_manager = cache  # This is the global cache manager
        # Use cache manager methods
        return f"Processing {data} with cache"
    
    result = process_with_cache("test_data")
    print(result)


# Example 7: Avoiding Service Locator Anti-pattern
def example_avoid_service_locator():
    """
    Show the difference between Service Locator (anti-pattern)
    and proper Dependency Injection.
    """
    from chiaroscuro_forge.di import get_container
    
    # ❌ BAD: Service Locator (anti-pattern)
    class BadProcessor:
        def process(self, data):
            # Hidden dependency - hard to test, unclear what's needed
            container = get_container()
            cache = container.get('cache')
            return cache.process(data)
    
    # ✅ GOOD: Dependency Injection
    class GoodProcessor:
        def __init__(self, cache):
            # Explicit dependency - easy to test, clear requirements
            self.cache = cache
        
        def process(self, data):
            return self.cache.process(data)
    
    print("Use GoodProcessor pattern for better testability!")


# Example 8: Integration with Existing Code
def example_integration():
    """Show how to integrate DI with existing chiaroscuro-forge code."""
    from chiaroscuro_forge.di import inject, get_container
    from chiaroscuro_forge.cache import get_cache_manager
    
    # Setup container with real services
    container = get_container()
    container.register('cache', get_cache_manager())
    
    # Create a wrapper function that uses DI
    @inject('cache')
    def cached_process_image(image_path, cache=None, **kwargs):
        """Process image with automatic caching."""
        # Check cache
        cache_key = f"process:{image_path}"
        cached_result = cache._stats_cache.get(cache_key, (None, None))[1]
        
        if cached_result:
            print(f"Cache hit for {image_path}")
            return cached_result
        
        # Process image (import here to avoid circular dependencies)
        from chiaroscuro_forge import process_image
        result = process_image(image_path, **kwargs)
        
        # Cache result
        cache._stats_cache[cache_key] = (None, result)
        
        return result
    
    print("Integration example ready!")


# Main example runner
if __name__ == '__main__':
    print("=" * 60)
    print("Chiaroscuro Forge - Dependency Injection Examples")
    print("=" * 60)
    
    print("\n1. Basic DI Container")
    print("-" * 60)
    example_basic_di()
    
    print("\n2. @inject Decorator")
    print("-" * 60)
    example_inject_decorator()
    
    print("\n3. Factory Pattern")
    print("-" * 60)
    example_factory_pattern()
    
    print("\n4. Testing with DI")
    print("-" * 60)
    example_testing_with_di()
    
    print("\n5. Constructor Injection")
    print("-" * 60)
    example_constructor_injection()
    
    print("\n6. Default Services")
    print("-" * 60)
    example_default_services()
    
    print("\n7. Avoid Service Locator Anti-pattern")
    print("-" * 60)
    example_avoid_service_locator()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
