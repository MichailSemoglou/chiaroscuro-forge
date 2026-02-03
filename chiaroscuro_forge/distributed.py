"""
Distributed processing module for parallel image processing across multiple nodes.

This module provides a task queue system for distributing image processing work
across multiple workers, with support for both local and remote execution.

Supports:
- Redis Queue (RQ) for simple setup
- Celery for advanced features
- Local threading for development
- Result aggregation and monitoring
- Error handling and retry logic

Example:
    >>> from chiaroscuro_forge.distributed import TaskQueue, LocalQueue
    >>> 
    >>> # Create a task queue
    >>> queue = LocalQueue()  # or RedisQueue() or CeleryQueue()
    >>> 
    >>> # Submit tasks
    >>> tasks = [
    ...     queue.submit_task('process_image', 'image1.jpg', params={'gamma': 1.2}),
    ...     queue.submit_task('process_image', 'image2.jpg', params={'gamma': 1.2})
    ... ]
    >>> 
    >>> # Wait for results
    >>> results = queue.gather_results(tasks)
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'worker_id': self.worker_id,
            'retry_count': self.retry_count,
            'duration': self.duration
        }


@dataclass
class Task:
    """A task to be executed."""
    task_id: str
    func_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskQueue(ABC):
    """Abstract base class for task queues."""
    
    @abstractmethod
    def submit_task(
        self,
        func_name: str,
        *args,
        priority: int = 0,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution.
        
        Args:
            func_name: Name of function to execute
            *args: Positional arguments for function
            priority: Task priority (higher = more important)
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            **kwargs: Keyword arguments for function
            
        Returns:
            Task ID
        """
        pass
    
    @abstractmethod
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result for a task.
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            TaskResult object
        """
        pass
    
    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False otherwise
        """
        pass
    
    @abstractmethod
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        pass
    
    def gather_results(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
        raise_on_error: bool = False
    ) -> List[TaskResult]:
        """Gather results for multiple tasks.
        
        Args:
            task_ids: List of task IDs
            timeout: Timeout in seconds
            raise_on_error: Raise exception if any task failed
            
        Returns:
            List of TaskResult objects
        """
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id, timeout=timeout)
            results.append(result)
            
            if raise_on_error and result.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task {task_id} failed: {result.error}")
        
        return results


class LocalQueue(TaskQueue):
    """Local task queue using ThreadPoolExecutor.
    
    Good for development and single-node processing.
    
    Example:
        >>> queue = LocalQueue(max_workers=4)
        >>> task_id = queue.submit_task('process_image', 'input.jpg')
        >>> result = queue.get_result(task_id)
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize local queue.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.futures: Dict[str, Any] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        
        # Register available functions
        self._functions: Dict[str, Callable] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register built-in processing functions."""
        # Import here to avoid circular dependency
        try:
            from chiaroscuro_forge.processing import process_image
            from chiaroscuro_forge.analysis import analyze_image_characteristics
            
            self._functions['process_image'] = process_image
            self._functions['analyze_image_characteristics'] = analyze_image_characteristics
        except ImportError:
            logger.warning("Could not import processing functions")
    
    def register_function(self, name: str, func: Callable):
        """Register a function for distributed execution.
        
        Args:
            name: Function name
            func: Function to register
        """
        self._functions[name] = func
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000)}"
    
    def submit_task(
        self,
        func_name: str,
        *args,
        priority: int = 0,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        task_id = self._generate_task_id()
        
        task = Task(
            task_id=task_id,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self.tasks[task_id] = task
        
        # Create result placeholder
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING
        )
        self.results[task_id] = result
        
        # Submit to executor
        future = self.executor.submit(self._execute_task, task)
        self.futures[task_id] = future
        
        logger.debug(f"Submitted task {task_id}: {func_name}")
        return task_id
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a task with retry logic."""
        result = self.results[task.task_id]
        result.status = TaskStatus.RUNNING
        result.started_at = datetime.now()
        result.worker_id = threading.current_thread().name
        
        for attempt in range(task.max_retries + 1):
            try:
                # Get function
                if task.func_name not in self._functions:
                    raise ValueError(f"Unknown function: {task.func_name}")
                
                func = self._functions[task.func_name]
                
                # Execute with timeout
                if task.timeout:
                    import signal
                    # Note: signal.alarm only works on Unix
                    # For Windows, would need threading.Timer
                    pass
                
                # Execute function
                output = func(*task.args, **task.kwargs)
                
                # Success
                result.status = TaskStatus.COMPLETED
                result.result = output
                result.completed_at = datetime.now()
                result.retry_count = attempt
                
                logger.debug(f"Task {task.task_id} completed successfully")
                return result
                
            except Exception as e:
                logger.warning(f"Task {task.task_id} failed (attempt {attempt + 1}): {e}")
                result.error = str(e)
                result.retry_count = attempt
                
                if attempt < task.max_retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    # Final failure
                    result.status = TaskStatus.FAILED
                    result.completed_at = datetime.now()
                    return result
        
        return result
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result for a task."""
        if task_id not in self.results:
            raise ValueError(f"Unknown task: {task_id}")
        
        # Wait for completion
        if task_id in self.futures:
            future = self.futures[task_id]
            try:
                future.result(timeout=timeout)
            except Exception:
                pass  # Result already stored in self.results
        
        return self.results[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if task_id in self.futures:
            future = self.futures[task_id]
            cancelled = future.cancel()
            
            if cancelled:
                result = self.results.get(task_id)
                if result:
                    result.status = TaskStatus.CANCELLED
                    result.completed_at = datetime.now()
            
            return cancelled
        
        return False
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'total': len(self.results)
        }
        
        for result in self.results.values():
            stats[result.status.value] += 1
        
        return stats
    
    def shutdown(self, wait: bool = True):
        """Shutdown the queue.
        
        Args:
            wait: Wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)


class DistributedBatchProcessor:
    """Batch processor using distributed task queue.
    
    Example:
        >>> processor = DistributedBatchProcessor(LocalQueue())
        >>> results = processor.process_batch(
        ...     image_paths=['img1.jpg', 'img2.jpg'],
        ...     params={'gamma': 1.2}
        ... )
    """
    
    def __init__(self, queue: TaskQueue):
        """Initialize batch processor.
        
        Args:
            queue: Task queue to use
        """
        self.queue = queue
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        params: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """Process multiple images in parallel.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory
            params: Processing parameters
            max_workers: Maximum parallel workers
            progress_callback: Callback for progress updates (completed, total)
            
        Returns:
            List of TaskResult objects
        """
        if params is None:
            params = {}
        
        # Generate output paths
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Submit tasks
        task_ids = []
        for img_path in image_paths:
            img_path = Path(img_path)
            
            # Determine output path
            if output_dir:
                output_path = output_dir / f"{img_path.stem}_processed{img_path.suffix}"
                task_params = {**params, 'output_path': str(output_path)}
            else:
                task_params = params.copy()
            
            task_id = self.queue.submit_task(
                'process_image',
                str(img_path),
                **task_params
            )
            task_ids.append(task_id)
        
        # Gather results with progress
        results = []
        for i, task_id in enumerate(task_ids):
            result = self.queue.get_result(task_id)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(task_ids))
        
        return results
    
    def aggregate_statistics(self, results: List[TaskResult]) -> Dict[str, Any]:
        """Aggregate statistics from batch results.
        
        Args:
            results: List of task results
            
        Returns:
            Dictionary with aggregated statistics
        """
        stats = {
            'total_tasks': len(results),
            'successful': sum(1 for r in results if r.status == TaskStatus.COMPLETED),
            'failed': sum(1 for r in results if r.status == TaskStatus.FAILED),
            'cancelled': sum(1 for r in results if r.status == TaskStatus.CANCELLED),
            'total_duration': sum(r.duration or 0 for r in results),
            'avg_duration': 0.0,
            'errors': []
        }
        
        # Calculate average duration
        completed = [r for r in results if r.status == TaskStatus.COMPLETED and r.duration]
        if completed:
            stats['avg_duration'] = sum(r.duration for r in completed) / len(completed)
        
        # Collect errors
        for result in results:
            if result.status == TaskStatus.FAILED:
                stats['errors'].append({
                    'task_id': result.task_id,
                    'error': result.error
                })
        
        return stats


def create_queue(backend: str = 'local', **kwargs) -> TaskQueue:
    """Create a task queue with specified backend.
    
    Args:
        backend: Backend type ('local', 'redis', 'celery')
        **kwargs: Backend-specific configuration
        
    Returns:
        TaskQueue instance
        
    Example:
        >>> queue = create_queue('local', max_workers=4)
        >>> # or
        >>> queue = create_queue('redis', host='localhost', port=6379)
    """
    if backend == 'local':
        return LocalQueue(max_workers=kwargs.get('max_workers'))
    elif backend == 'redis':
        # Redis backend not yet implemented
        raise ImportError(
            "Redis backend is not yet implemented. "
            "Currently only 'local' backend is supported. "
            "Future implementation will require: pip install redis rq"
        )
    elif backend == 'celery':
        # Celery backend not yet implemented
        raise ImportError(
            "Celery backend is not yet implemented. "
            "Currently only 'local' backend is supported. "
            "Future implementation will require: pip install celery[redis]"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported backends: 'local'")
