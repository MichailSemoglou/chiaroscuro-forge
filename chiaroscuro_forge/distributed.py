"""
Distributed processing module for parallel image processing across multiple nodes.

This module provides a task queue system for distributing image processing work
across multiple workers, with support for both local and remote execution.

Supports:
- Redis Queue (RQ) for simple setup
- Celery for advanced features
- Local threading for development
- Result aggregation and monitoring
- Error handling and retry logic with jitter
- Queue health and observability
- Context-manager lifecycle

Example:
    >>> from chiaroscuro_forge.distributed import TaskQueue, LocalQueue
    >>>
    >>> with LocalQueue(max_workers=4) as queue:
    ...     task_id = queue.submit_task('process_image', 'image1.jpg',
    ...                                gamma_correction=1.2)
    ...     result = queue.get_result(task_id)
    ...     print(result.status)
"""

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
)
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)
"""Exception types that trigger automatic retry. Others are treated as fatal."""

MAX_BACKOFF_DELAY = 30.0
"""Maximum backoff delay in seconds for retry sleeps."""


@dataclass
class QueueConfig:
    """Lifecycle configuration for a task queue.

    Parameters
    ----------
    max_workers :
        Maximum number of worker threads (``None`` = auto, positive integer otherwise).
    max_retries :
        Default maximum retry attempts per task (must be non-negative).
    retry_delay_base :
        Base delay in seconds for exponential backoff (must be non-negative).
    retry_jitter :
        Maximum random jitter added to backoff delay (seconds, must be non-negative).
    task_timeout :
        Default per-task timeout in seconds (``None`` = no timeout, positive otherwise).
    ttl_seconds :
        Time-to-live for completed task results. Tasks older than this
        are eligible for cleanup (must be positive). Default: 1 hour.
    """

    max_workers: Optional[int] = None
    max_retries: int = 3
    retry_delay_base: float = 2.0
    retry_jitter: float = 0.5
    task_timeout: Optional[float] = None
    ttl_seconds: float = 3600.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer or None")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay_base < 0:
            raise ValueError("retry_delay_base must be non-negative")
        if self.retry_jitter < 0:
            raise ValueError("retry_jitter must be non-negative")
        if self.task_timeout is not None and self.task_timeout <= 0:
            raise ValueError("task_timeout must be positive or None")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")


@dataclass
class TaskResult:
    """Result of a distributed task."""

    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    retry_count: int = 0

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def wait_time(self) -> Optional[float]:
        if self.created_at and self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "worker_id": self.worker_id,
            "retry_count": self.retry_count,
            "duration": self.duration,
            "wait_time": self.wait_time,
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


@dataclass
class QueueHealth:
    """Snapshot of queue health and statistics."""

    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    total: int = 0
    avg_duration: Optional[float] = None
    avg_wait_time: Optional[float] = None
    latest_error: Optional[str] = None
    worker_count: int = 0


class TaskQueue(ABC):
    """Abstract base class for task queues."""

    @abstractmethod
    def submit_task(
        self,
        func_name: str,
        *args,
        priority: int = 0,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        pass

    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        pass

    @abstractmethod
    def get_health(self) -> QueueHealth:
        pass

    @abstractmethod
    def close(self) -> None:
        """Shut down the queue, cancelling pending tasks and releasing resources."""
        pass

    def shutdown(self, wait: bool = True) -> None:
        """Deprecated: use ``close()`` instead."""
        self.close()

    def get_queue_stats(self) -> Dict[str, Any]:
        """Deprecated: use ``get_health()`` instead."""
        health = self.get_health()
        return {
            "total": health.total,
            "completed": health.completed,
            "failed": health.failed,
            "pending": health.pending,
            "running": health.running,
            "cancelled": health.cancelled,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def gather_results(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
        raise_on_error: bool = False,
    ) -> List[TaskResult]:
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id, timeout=timeout)
            results.append(result)
            if raise_on_error and result.status in {
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
                TaskStatus.CANCELLED,
            }:
                raise RuntimeError(f"Task {task_id} failed: {result.error}")
        return results


def _backoff_delay(attempt: int, base: float = 2.0, jitter: float = 0.5) -> float:
    """Exponential backoff with random jitter.

    Parameters
    ----------
    attempt :
        Zero-based retry attempt number.
    base :
        Base delay in seconds.
    jitter :
        Maximum jitter in seconds.

    Returns
    -------
    float
        Seconds to wait before the next retry.
    """
    capped_delay = min(base**attempt, MAX_BACKOFF_DELAY)
    # Non-security jitter for retry pacing uses Python's random module.
    jitter_amount = random.uniform(0, jitter)  # noqa: S311
    return capped_delay + jitter_amount


class LocalQueue(TaskQueue):
    """Local task queue using ``ThreadPoolExecutor``.

    Good for development and single-node processing. Supports context
    manager protocol for automatic cleanup.

    Example
    -------
    >>> with LocalQueue(max_workers=4) as queue:
    ...     task_id = queue.submit_task('process_image', 'input.jpg')
    ...     result = queue.get_result(task_id)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        config: Optional[QueueConfig] = None,
    ):
        if config is None:
            config = QueueConfig(max_workers=max_workers)
        else:
            config._validate()
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.worker_count = config.max_workers or self.executor._max_workers
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.futures: Dict[str, Any] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._closed = False
        self._functions: Dict[str, Callable] = {}
        self._register_builtin_functions()
        logger.info(
            "LocalQueue started (max_workers=%s)",
            config.max_workers,
        )

    def _register_builtin_functions(self):
        try:
            from chiaroscuro_forge.analysis import analyze_image_characteristics
            from chiaroscuro_forge.processing import process_image

            self._functions["process_image"] = process_image
            self._functions["analyze_image_characteristics"] = analyze_image_characteristics
        except ImportError:
            logger.warning("Could not import processing functions")

    def register_function(self, name: str, func: Callable):
        self._functions[name] = func

    def _generate_task_id(self) -> str:
        self._task_counter += 1
        return f"task_{self._task_counter}_{int(time.time() * 1000)}"

    def submit_task(
        self,
        func_name: str,
        *args,
        priority: int = 0,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot submit tasks to a closed queue")

            if max_retries is None:
                max_retries = self.config.max_retries
            if timeout is None:
                timeout = self.config.task_timeout

            task_id = self._generate_task_id()
            task = Task(
                task_id=task_id,
                func_name=func_name,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
            )
            self.tasks[task_id] = task
            self.results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )
            try:
                future = self.executor.submit(self._execute_task, task)
            except Exception:
                self.tasks.pop(task_id, None)
                self.results.pop(task_id, None)
                raise
            self.futures[task_id] = future
            logger.debug("Submitted task %s: %s(*%r, **%r)", task_id, func_name, args, kwargs)
            return task_id

    def _is_terminal_status(self, status: Optional[TaskStatus]) -> bool:
        return status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        }

    def _execute_task(self, task: Task) -> TaskResult:
        with self._lock:
            result = self.results[task.task_id]
            if self._is_terminal_status(result.status):
                return result
            result.status = TaskStatus.RUNNING
            result.started_at = datetime.now()
            result.worker_id = threading.current_thread().name

        for attempt in range(task.max_retries + 1):
            try:
                if task.func_name not in self._functions:
                    raise ValueError(f"Unknown function: {task.func_name}")
                func = self._functions[task.func_name]
                output = func(*task.args, **task.kwargs)
                with self._lock:
                    result = self.results[task.task_id]
                    if self._is_terminal_status(result.status):
                        return result
                    result.result = output
                    result.retry_count = attempt
                    result.completed_at = datetime.now()
                    result.status = TaskStatus.COMPLETED
                logger.debug("Task %s completed (attempt %d)", task.task_id, attempt + 1)
                return result
            except RETRYABLE_ERRORS as exc:
                if attempt < task.max_retries:
                    delay = _backoff_delay(
                        attempt,
                        base=self.config.retry_delay_base,
                        jitter=self.config.retry_jitter,
                    )
                    logger.warning(
                        "Task %s failed (retryable, attempt %d/%d): %s. " "Retrying in %.1fs.",
                        task.task_id,
                        attempt + 1,
                        task.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    with self._lock:
                        result = self.results[task.task_id]
                        result.retry_count = attempt + 1
                else:
                    with self._lock:
                        result = self.results[task.task_id]
                        if self._is_terminal_status(result.status):
                            return result
                        result.error = str(exc)
                        result.error_type = type(exc).__name__
                        result.completed_at = datetime.now()
                        result.status = TaskStatus.FAILED
                    logger.error(
                        "Task %s failed after %d retries: %s",
                        task.task_id,
                        task.max_retries,
                        exc,
                    )
                    return result
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    result = self.results[task.task_id]
                    if self._is_terminal_status(result.status):
                        return result
                    result.error = str(exc)
                    result.error_type = type(exc).__name__
                    result.completed_at = datetime.now()
                    result.status = TaskStatus.FAILED
                logger.error("Task %s failed (fatal): %s", task.task_id, exc)
                return result

        return result

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        with self._lock:
            if task_id not in self.results:
                raise ValueError(f"Unknown task: {task_id}")
            task = self.tasks.get(task_id)
            effective_timeout = timeout
            if effective_timeout is None and task is not None:
                effective_timeout = task.timeout
            if effective_timeout is None:
                effective_timeout = self.config.task_timeout

        if task_id in self.futures:
            future = self.futures[task_id]
            try:
                future.result(timeout=effective_timeout)
            except FuturesTimeoutError:
                with self._lock:
                    result = self.results.get(task_id)
                    if result and not self._is_terminal_status(result.status):
                        result.status = TaskStatus.TIMED_OUT
                        result.completed_at = datetime.now()
                        result.error = f"Task timed out after {effective_timeout} seconds"
                        result.error_type = FuturesTimeoutError.__name__
                logger.warning("Task %s timed out after %.2fs", task_id, effective_timeout)
                return self.results[task_id]
            except CancelledError:
                with self._lock:
                    result = self.results.get(task_id)
                    if result and not self._is_terminal_status(result.status):
                        result.status = TaskStatus.CANCELLED
                        result.completed_at = datetime.now()
                logger.info("Task %s cancelled", task_id)
                return self.results[task_id]
            except Exception:
                raise

        with self._lock:
            return self.results[task_id]

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self.futures:
                return False
            future = self.futures[task_id]
            cancelled = future.cancel()
            if cancelled:
                result = self.results.get(task_id)
                if result and not self._is_terminal_status(result.status):
                    result.status = TaskStatus.CANCELLED
                    result.completed_at = datetime.now()
            if cancelled:
                logger.info("Task %s cancelled", task_id)
            return bool(cancelled)

    def get_health(self) -> QueueHealth:
        health = QueueHealth(worker_count=self.worker_count)
        durations = []
        wait_times = []
        latest_error = None
        latest_error_completed_at = None
        with self._lock:
            for result in self.results.values():
                health.total += 1
                status_key = result.status.value
                current = getattr(health, status_key, 0)
                setattr(health, status_key, current + 1)
                if result.duration is not None:
                    durations.append(result.duration)
                if result.wait_time is not None:
                    wait_times.append(result.wait_time)
                if result.error and (
                    latest_error is None
                    or (
                        result.completed_at is not None
                        and (
                            latest_error_completed_at is None
                            or result.completed_at > latest_error_completed_at
                        )
                    )
                ):
                    latest_error = result.error
                    latest_error_completed_at = result.completed_at
        if durations:
            health.avg_duration = sum(durations) / len(durations)
        if wait_times:
            health.avg_wait_time = sum(wait_times) / len(wait_times)
        health.latest_error = latest_error
        return health

    def purge_completed(self) -> int:
        """Remove completed/cancelled/failed tasks older than the TTL.

        Returns
        -------
        int
            Number of tasks removed.
        """
        cutoff = datetime.now() - timedelta(seconds=self.config.ttl_seconds)
        removed = 0
        with self._lock:
            stale = [
                tid
                for tid, r in self.results.items()
                if r.status
                in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.TIMED_OUT,
                )
                and r.completed_at is not None
                and r.completed_at < cutoff
            ]
            for tid in stale:
                del self.results[tid]
                if tid in self.futures:
                    del self.futures[tid]
                if tid in self.tasks:
                    del self.tasks[tid]
                removed += 1
        if removed:
            logger.info("Purged %d completed tasks", removed)
        return removed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            logger.info("LocalQueue shutting down…")

            shutdown_futures = []
            for task_id, future in list(self.futures.items()):
                if not future.done():
                    shutdown_futures.append((task_id, future))

            for task_id, future in shutdown_futures:
                if future.cancel():
                    result = self.results.get(task_id)
                    if result and not self._is_terminal_status(result.status):
                        result.status = TaskStatus.CANCELLED
                        result.completed_at = datetime.now()
                    logger.info("Task %s cancelled during shutdown", task_id)

        self.executor.shutdown(wait=False)
        self.purge_completed()
        logger.info("LocalQueue closed")


class DistributedBatchProcessor:
    """Batch processor using distributed task queue.

    Example
    -------
    >>> with LocalQueue(max_workers=4) as q:
    ...     processor = DistributedBatchProcessor(q)
    ...     results = processor.process_batch(
    ...         image_paths=['img1.jpg', 'img2.jpg'],
    ...         params={'gamma': 1.2},
    ...     )
    """

    def __init__(self, queue: TaskQueue):
        self.queue = queue

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        if params is None:
            params = {}
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        task_ids = []
        for img_path in image_paths:
            img_path = Path(img_path)
            task_params = dict(params)
            if output_dir:
                task_params["output_path"] = str(
                    output_dir / f"{img_path.stem}_processed{img_path.suffix}"
                )
            task_id = self.queue.submit_task(
                "process_image",
                str(img_path),
                **task_params,
            )
            task_ids.append(task_id)

        results = []
        for i, task_id in enumerate(task_ids):
            result = self.queue.get_result(task_id, timeout=timeout)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(task_ids))

        return results

    def aggregate_statistics(self, results: List[TaskResult]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total_tasks": len(results),
            "successful": sum(1 for r in results if r.status == TaskStatus.COMPLETED),
            "failed": sum(1 for r in results if r.status == TaskStatus.FAILED),
            "cancelled": sum(1 for r in results if r.status == TaskStatus.CANCELLED),
            "total_duration": sum(float(r.duration or 0.0) for r in results),
            "avg_duration": 0.0,
            "errors": [],
        }
        completed = [r for r in results if r.status == TaskStatus.COMPLETED and r.duration is not None]
        if completed:
            duration_values = [float(r.duration) for r in completed if r.duration is not None]
            stats["avg_duration"] = sum(duration_values) / len(duration_values)
        for result in results:
            if result.status == TaskStatus.FAILED:
                stats["errors"].append(
                    {
                        "task_id": result.task_id,
                        "error": result.error,
                        "error_type": result.error_type,
                    }
                )
        return stats


def create_queue(backend: str = "local", **kwargs) -> TaskQueue:
    """Create a task queue with specified backend.

    Parameters
    ----------
    backend :
        Backend type (``"local"``, ``"redis"``, ``"celery"``).
    **kwargs :
        Backend-specific configuration for the local backend. Accepted keys are
        ``max_workers``, ``max_retries``, ``retry_delay_base``,
        ``retry_jitter``, ``task_timeout``, and ``ttl_seconds``.

    Returns
    -------
    TaskQueue

    Raises
    ------
    ImportError
        If the backend's optional dependency is not installed.
    ValueError
        If *backend* is unknown.

    Example
    -------
    >>> queue = create_queue("local", max_workers=4)
    """
    if backend == "local":
        accepted_fields = set(QueueConfig.__dataclass_fields__)
        unknown_fields = set(kwargs) - accepted_fields
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise TypeError(f"create_queue() got unexpected keyword arguments: {unknown}")
        config = QueueConfig(**kwargs)
        return LocalQueue(config=config)
    elif backend == "redis":
        raise ImportError(
            "Redis backend is not yet implemented. "
            "Install dependencies with: pip install redis rq"
        )
    elif backend == "celery":
        raise ImportError(
            "Celery backend is not yet implemented. "
            "Install dependencies with: pip install celery[redis]"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported backends: 'local'")
