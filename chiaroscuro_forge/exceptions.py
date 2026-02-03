"""
Custom Exceptions for Chiaroscuro Forge

This module defines custom exception classes used throughout the package.
"""


class ImageProcessingError(Exception):
    """
    Base exception class for image processing errors.
    
    This exception is raised when any image processing operation fails,
    including validation errors, processing failures, or I/O errors.
    
    Examples
    --------
    >>> from chiaroscuro_forge import process_image
    >>> try:
    ...     process_image("nonexistent.jpg")
    ... except ImageProcessingError as e:
    ...     print(f"Processing failed: {e}")
    """
    pass
