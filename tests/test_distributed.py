"""Tests for distributed processing module."""

import threading
import unittest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from chiaroscuro_forge.distributed import (
    TaskStatus,
    TaskResult,
    Task,
    TaskQueue,
    QueueConfig,
    QueueHealth,
    LocalQueue,
    DistributedBatchProcessor,
    create_queue,
    MAX_BACKOFF_DELAY,
    _backoff_delay,
)


class TestTaskStatus(unittest.TestCase):
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test task status enum values."""
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.RUNNING.value, "running")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.FAILED.value, "failed")
        self.assertEqual(TaskStatus.CANCELLED.value, "cancelled")
        self.assertEqual(TaskStatus.TIMED_OUT.value, "timed_out")


class TestTaskResult(unittest.TestCase):
    """Tests for TaskResult dataclass."""

    def test_create_task_result(self):
        """Test creating a task result."""
        result = TaskResult(
            task_id="task_123", status=TaskStatus.COMPLETED, result={"output": "test.jpg"}
        )

        self.assertEqual(result.task_id, "task_123")
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(result.result)

    def test_duration_calculation(self):
        """Test duration calculation."""
        from datetime import datetime, timedelta

        result = TaskResult(
            task_id="task_123",
            status=TaskStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 10, 0, 5),
        )

        self.assertEqual(result.duration, 5.0)

    def test_duration_none_when_not_complete(self):
        """Test duration is None when task not complete."""
        from datetime import datetime

        result = TaskResult(
            task_id="task_123", status=TaskStatus.RUNNING, started_at=datetime.now()
        )

        self.assertIsNone(result.duration)

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = TaskResult(task_id="task_123", status=TaskStatus.COMPLETED, result="success")

        data = result.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(
            set(data),
            {
                "task_id",
                "status",
                "result",
                "error",
                "error_type",
                "started_at",
                "completed_at",
                "created_at",
                "worker_id",
                "retry_count",
                "duration",
                "wait_time",
            },
        )
        self.assertEqual(data["task_id"], "task_123")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["error_type"], None)
        self.assertEqual(data["created_at"], None)
        self.assertEqual(data["wait_time"], None)
        self.assertEqual(data["duration"], None)


class TestTask(unittest.TestCase):
    """Tests for Task dataclass."""

    def test_create_task(self):
        """Test creating a task."""
        task = Task(
            task_id="task_123",
            func_name="process_image",
            args=("input.jpg",),
            kwargs={"gamma": 1.2},
        )

        self.assertEqual(task.task_id, "task_123")
        self.assertEqual(task.func_name, "process_image")
        self.assertEqual(task.args, ("input.jpg",))
        self.assertEqual(task.kwargs, {"gamma": 1.2})

    def test_default_values(self):
        """Test task default values."""
        task = Task(task_id="task_123", func_name="test_func")

        self.assertEqual(task.args, ())
        self.assertEqual(task.kwargs, {})
        self.assertEqual(task.priority, 0)
        self.assertEqual(task.max_retries, 3)
        self.assertIsNone(task.timeout)


class TestBackoffDelay(unittest.TestCase):
    """Tests for exponential backoff helper."""

    def test_backoff_delay_is_capped_before_jitter(self):
        """Backoff should not grow without bound before jitter is added."""
        with patch("chiaroscuro_forge.distributed.random.uniform", return_value=0.0):
            delay = _backoff_delay(20, base=2.0, jitter=0.5)

        self.assertEqual(delay, MAX_BACKOFF_DELAY)


class TestLocalQueue(unittest.TestCase):
    """Tests for LocalQueue."""

    def setUp(self):
        """Set up test fixtures."""
        self.queue = LocalQueue(max_workers=2)

    def tearDown(self):
        """Clean up after tests."""
        self.queue.close()

    def test_create_queue(self):
        """Test creating a local queue."""
        queue = LocalQueue(max_workers=4)
        self.assertIsNotNone(queue)
        self.assertEqual(queue.config.max_workers, 4)
        queue.close()

    def test_register_function(self):
        """Test registering a custom function."""

        def custom_func(x):
            return x * 2

        self.queue.register_function("custom", custom_func)
        self.assertIn("custom", self.queue._functions)

    def test_submit_task(self):
        """Test submitting a task."""

        def test_func():
            return "success"

        self.queue.register_function("test_func", test_func)
        task_id = self.queue.submit_task("test_func")

        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.queue.tasks)

    def test_get_result(self):
        """Test getting task result."""

        def add(a, b):
            return a + b

        self.queue.register_function("add", add)
        task_id = self.queue.submit_task("add", 2, 3)

        result = self.queue.get_result(task_id, timeout=5)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, 5)

    def test_get_result_with_kwargs(self):
        """Test task execution with keyword arguments."""

        def multiply(a, b):
            return a * b

        self.queue.register_function("multiply", multiply)
        task_id = self.queue.submit_task("multiply", a=3, b=4)

        result = self.queue.get_result(task_id, timeout=5)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, 12)

    def test_task_failure(self):
        """Test task failure handling."""

        def failing_func():
            raise RuntimeError("Test error")

        self.queue.register_function("failing_func", failing_func)
        task_id = self.queue.submit_task("failing_func", max_retries=0)

        result = self.queue.get_result(task_id, timeout=5)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("Test error", result.error)

    def test_task_retry(self):
        """Test task retry logic."""
        queue = LocalQueue(
            max_workers=2,
            config=QueueConfig(max_retries=3, retry_delay_base=0.0, retry_jitter=0.0),
        )
        self.addCleanup(queue.close)
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Temporary error")
            return "success"

        queue.register_function("flaky_func", flaky_func)
        task_id = queue.submit_task("flaky_func")

        with patch("chiaroscuro_forge.distributed._backoff_delay", return_value=0.0):
            result = queue.get_result(task_id, timeout=10)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "success")
        self.assertGreater(result.retry_count, 0)

    def test_fatal_error_skips_retries(self):
        """Fatal errors should fail immediately without retrying."""
        queue = LocalQueue(
            max_workers=2,
            config=QueueConfig(max_retries=3, retry_delay_base=0.0, retry_jitter=0.0),
        )
        self.addCleanup(queue.close)
        call_count = [0]

        def fatal_func():
            call_count[0] += 1
            raise RuntimeError("Fatal error")

        queue.register_function("fatal_func", fatal_func)
        task_id = queue.submit_task("fatal_func")

        with patch("chiaroscuro_forge.distributed._backoff_delay", return_value=0.0):
            result = queue.get_result(task_id, timeout=5)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertEqual(call_count[0], 1)
        self.assertIn("Fatal error", result.error)

    def test_cancel_task(self):
        """Test cancelling a queued task."""

        def slow_func():
            time.sleep(10)
            return "done"

        release = threading.Event()
        started = threading.Event()
        started_count = [0]

        def blocker():
            with self.queue._lock:
                started_count[0] += 1
                if started_count[0] == 2:
                    started.set()
            release.wait(timeout=5)
            return "blocked"

        self.queue.register_function("slow_func", slow_func)
        self.queue.register_function("blocker", blocker)

        first_blocker = self.queue.submit_task("blocker")
        second_blocker = self.queue.submit_task("blocker")
        self.assertTrue(started.wait(timeout=1))
        task_id = self.queue.submit_task("slow_func")

        cancelled = self.queue.cancel_task(task_id)
        release.set()
        self.queue.get_result(first_blocker, timeout=5)
        self.queue.get_result(second_blocker, timeout=5)

        self.assertTrue(cancelled)

    def test_get_result_times_out_using_task_timeout(self):
        """Tasks should report TIMED_OUT when they exceed their configured timeout."""

        queue = LocalQueue(max_workers=1, config=QueueConfig(task_timeout=0.1))
        self.addCleanup(queue.close)

        def slow_func():
            time.sleep(0.3)
            return "done"

        queue.register_function("slow_func", slow_func)
        task_id = queue.submit_task("slow_func")

        result = queue.get_result(task_id)

        self.assertEqual(result.status, TaskStatus.TIMED_OUT)

    def test_wait_timeout_is_not_overwritten_by_later_completion(self):
        """A wait timeout should remain the terminal outcome even if the task finishes later."""

        queue = LocalQueue(max_workers=1)
        self.addCleanup(queue.close)

        def slow_func():
            time.sleep(0.2)
            return "done"

        queue.register_function("slow_func", slow_func)
        task_id = queue.submit_task("slow_func")

        result = queue.get_result(task_id, timeout=0.05)
        self.assertEqual(result.status, TaskStatus.TIMED_OUT)
        self.assertIn("timed out", result.error.lower())
        self.assertEqual(result.error_type, TimeoutError.__name__)

        time.sleep(0.25)
        final_result = queue.get_result(task_id, timeout=0.1)

        self.assertEqual(final_result.status, TaskStatus.TIMED_OUT)
        self.assertEqual(final_result.error, result.error)
        self.assertEqual(final_result.error_type, result.error_type)

    def test_close_cancels_pending_futures(self):
        """Closing the queue should cancel pending tasks without waiting for them."""

        queue = LocalQueue(max_workers=1)
        self.addCleanup(queue.close)

        started = threading.Event()
        release = threading.Event()

        def blocking_func():
            started.set()
            release.wait(timeout=5)
            return "done"

        queue.register_function("blocking_func", blocking_func)
        queue.submit_task("blocking_func")
        self.assertTrue(started.wait(timeout=1))

        pending_task_id = queue.submit_task("blocking_func")
        queue.close()
        release.set()

        result = queue.get_result(pending_task_id, timeout=0.1)
        self.assertEqual(result.status, TaskStatus.CANCELLED)

    def test_get_queue_health(self):
        """Test getting queue health."""

        def quick_func():
            return "done"

        self.queue.register_function("quick_func", quick_func)

        # Submit multiple tasks
        task_ids = [self.queue.submit_task("quick_func") for _ in range(3)]

        # Wait for completion
        for task_id in task_ids:
            self.queue.get_result(task_id, timeout=5)

        health = self.queue.get_health()

        self.assertEqual(health.total, 3)
        self.assertEqual(health.completed, 3)

    def test_get_health_returns_pending_count(self):
        """Test health when tasks are still pending."""

        def slow_func():
            time.sleep(0.5)
            return "done"

        self.queue.register_function("slow_func", slow_func)
        task_id = self.queue.submit_task("slow_func")

        health = self.queue.get_health()
        self.assertGreater(health.total, 0)
        self.assertEqual(health.pending + health.running, 1)

        self.queue.get_result(task_id, timeout=5)

    def test_gather_results(self):
        """Test gathering multiple results."""

        def square(x):
            return x**2

        self.queue.register_function("square", square)

        task_ids = [self.queue.submit_task("square", i) for i in range(5)]

        results = self.queue.gather_results(task_ids, timeout=5)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.status == TaskStatus.COMPLETED for r in results))
        self.assertEqual([r.result for r in results], [0, 1, 4, 9, 16])

    def test_gather_results_with_error(self):
        """Test gathering results with raise_on_error."""

        def conditional_func(x):
            if x == 2:
                raise RuntimeError("Error for 2")
            return x

        self.queue.register_function("conditional_func", conditional_func)

        task_ids = [self.queue.submit_task("conditional_func", i, max_retries=0) for i in range(4)]

        with self.assertRaises(RuntimeError):
            self.queue.gather_results(task_ids, timeout=5, raise_on_error=True)

    def test_unknown_function(self):
        """Test submitting task with unknown function."""
        task_id = self.queue.submit_task("nonexistent_func", max_retries=0)
        result = self.queue.get_result(task_id, timeout=5)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("Unknown function", result.error)

    def test_parallel_execution(self):
        """Test parallel task execution."""

        def timed_func(duration):
            time.sleep(duration)
            return duration

        self.queue.register_function("timed_func", timed_func)

        # Submit 4 tasks with 0.5s each to 2 workers
        # Should take ~1s total (2 parallel batches)
        start_time = time.time()
        task_ids = [self.queue.submit_task("timed_func", 0.5) for _ in range(4)]

        results = self.queue.gather_results(task_ids, timeout=10)
        total_time = time.time() - start_time

        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.status == TaskStatus.COMPLETED for r in results))
        # Should complete in ~1s (parallel), not 2s (sequential)
        self.assertLess(total_time, 1.8)

    def test_context_manager(self):
        """Test using LocalQueue as a context manager."""

        def quick_func(x):
            return x * 2

        with LocalQueue(max_workers=2) as q:
            q.register_function("quick_func", quick_func)
            task_id = q.submit_task("quick_func", 5)
            result = q.get_result(task_id, timeout=5)
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(result.result, 10)

    def test_purge_completed(self):
        """Test purging completed tasks (default TTL keeps fresh tasks)."""

        def quick_func():
            return "done"

        self.queue.register_function("quick_func", quick_func)
        task_ids = [self.queue.submit_task("quick_func") for _ in range(3)]
        for tid in task_ids:
            self.queue.get_result(tid, timeout=5)

        # Fresh tasks are not purged under default TTL (1 hour)
        removed = self.queue.purge_completed()
        self.assertEqual(removed, 0)

        # With very short TTL, tasks should be removed
        self.queue.config.ttl_seconds = -1
        removed = self.queue.purge_completed()
        self.assertEqual(removed, 3)

    def test_submit_task_cleans_up_when_submit_fails(self):
        """Rejected submissions should not leave pending task bookkeeping behind."""
        queue = LocalQueue(max_workers=1)
        self.addCleanup(queue.close)

        with patch.object(queue.executor, "submit", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                queue.submit_task("any_func")

        self.assertEqual(queue.tasks, {})
        self.assertEqual(queue.results, {})
        self.assertEqual(queue.futures, {})

    def test_submit_to_closed_queue_raises(self):
        """Test submitting to a closed queue raises RuntimeError."""
        self.queue.close()
        with self.assertRaises(RuntimeError):
            self.queue.submit_task("any_func")


class TestDistributedBatchProcessor(unittest.TestCase):
    """Tests for DistributedBatchProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.queue = LocalQueue(max_workers=2)
        self.processor = DistributedBatchProcessor(self.queue)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        self.queue.close()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_image(self, name: str) -> Path:
        """Create a test image."""
        img_path = Path(self.temp_dir) / name
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)
        return img_path

    def test_create_processor(self):
        """Test creating a batch processor."""
        processor = DistributedBatchProcessor(self.queue)
        self.assertIsNotNone(processor)
        self.assertEqual(processor.queue, self.queue)

    def test_aggregate_statistics(self):
        """Test aggregating statistics from results."""
        from datetime import datetime, timedelta

        results = [
            TaskResult(
                task_id="task_1",
                status=TaskStatus.COMPLETED,
                started_at=datetime.now(),
                completed_at=datetime.now() + timedelta(seconds=1),
            ),
            TaskResult(
                task_id="task_2",
                status=TaskStatus.COMPLETED,
                started_at=datetime.now(),
                completed_at=datetime.now() + timedelta(seconds=2),
            ),
            TaskResult(task_id="task_3", status=TaskStatus.FAILED, error="Test error"),
        ]

        stats = self.processor.aggregate_statistics(results)

        self.assertEqual(stats["total_tasks"], 3)
        self.assertEqual(stats["successful"], 2)
        self.assertEqual(stats["failed"], 1)
        self.assertGreater(stats["avg_duration"], 0)
        self.assertEqual(len(stats["errors"]), 1)
        self.assertEqual(stats["errors"][0]["error_type"], None)

    def test_process_batch_writes_outputs(self):
        """process_batch should write processed output files and report progress."""
        from chiaroscuro_forge.processing import process_image

        self.queue.register_function("process_image", process_image)

        image_one = self._create_test_image("input_one.png")
        image_two = self._create_test_image("input_two.png")
        progress_updates = []

        results = self.processor.process_batch(
            image_paths=[image_one, image_two],
            output_dir=self.temp_dir,
            params={"gamma_correction": 1.2, "calculate_metrics": False},
            progress_callback=lambda current, total: progress_updates.append((current, total)),
            timeout=5,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.status == TaskStatus.COMPLETED for result in results))
        self.assertEqual(progress_updates, [(1, 2), (2, 2)])

        expected_one = Path(self.temp_dir) / "input_one_processed.png"
        expected_two = Path(self.temp_dir) / "input_two_processed.png"
        self.assertTrue(expected_one.exists())
        self.assertTrue(expected_two.exists())


class TestCreateQueue(unittest.TestCase):
    """Tests for create_queue factory function."""

    def test_create_local_queue(self):
        """Test creating local queue."""
        queue = create_queue("local", max_workers=4)

        self.assertIsInstance(queue, LocalQueue)
        self.assertEqual(queue.config.max_workers, 4)
        queue.close()

    def test_create_local_queue_with_full_config(self):
        """Local queue factory should pass through all supported QueueConfig fields."""
        queue = create_queue(
            "local",
            max_workers=2,
            max_retries=5,
            retry_delay_base=3.0,
            retry_jitter=0.75,
            task_timeout=1.5,
            ttl_seconds=600.0,
        )

        self.assertEqual(queue.config.max_workers, 2)
        self.assertEqual(queue.config.max_retries, 5)
        self.assertEqual(queue.config.retry_delay_base, 3.0)
        self.assertEqual(queue.config.retry_jitter, 0.75)
        self.assertEqual(queue.config.task_timeout, 1.5)
        self.assertEqual(queue.config.ttl_seconds, 600.0)
        queue.close()

    def test_create_local_queue_rejects_unknown_kwargs(self):
        """Local queue factory should reject unknown configuration keys."""
        with self.assertRaises(TypeError):
            create_queue("local", max_workers=2, unexpected_option=True)

    def test_create_local_queue_rejects_invalid_worker_count(self):
        """Factory creation should reject non-positive worker counts."""
        with self.assertRaises(ValueError):
            create_queue("local", max_workers=0)

    def test_direct_local_queue_rejects_invalid_timeout(self):
        """Direct construction should reject non-positive task timeouts."""
        with self.assertRaises(ValueError):
            LocalQueue(config=QueueConfig(task_timeout=0.0))

    def test_create_unknown_backend(self):
        """Test error for unknown backend."""
        with self.assertRaises(ValueError):
            create_queue("nonexistent")

    def test_create_redis_queue_without_dependency(self):
        """Test redis queue without redis installed."""
        with self.assertRaises(ImportError):
            create_queue("redis")

    def test_create_celery_queue_without_dependency(self):
        """Test celery queue without celery installed."""
        with self.assertRaises(ImportError):
            create_queue("celery")


if __name__ == "__main__":
    unittest.main()
