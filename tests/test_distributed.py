"""Tests for distributed processing module."""

import unittest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from chiaroscuro_forge.distributed import (
    TaskStatus, TaskResult, Task, TaskQueue,
    LocalQueue, DistributedBatchProcessor, create_queue
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


class TestTaskResult(unittest.TestCase):
    """Tests for TaskResult dataclass."""
    
    def test_create_task_result(self):
        """Test creating a task result."""
        result = TaskResult(
            task_id="task_123",
            status=TaskStatus.COMPLETED,
            result={"output": "test.jpg"}
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
            completed_at=datetime(2024, 1, 1, 10, 0, 5)
        )
        
        self.assertEqual(result.duration, 5.0)
    
    def test_duration_none_when_not_complete(self):
        """Test duration is None when task not complete."""
        from datetime import datetime
        
        result = TaskResult(
            task_id="task_123",
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        self.assertIsNone(result.duration)
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = TaskResult(
            task_id="task_123",
            status=TaskStatus.COMPLETED,
            result="success"
        )
        
        data = result.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data['task_id'], "task_123")
        self.assertEqual(data['status'], "completed")


class TestTask(unittest.TestCase):
    """Tests for Task dataclass."""
    
    def test_create_task(self):
        """Test creating a task."""
        task = Task(
            task_id="task_123",
            func_name="process_image",
            args=("input.jpg",),
            kwargs={"gamma": 1.2}
        )
        
        self.assertEqual(task.task_id, "task_123")
        self.assertEqual(task.func_name, "process_image")
        self.assertEqual(task.args, ("input.jpg",))
        self.assertEqual(task.kwargs, {"gamma": 1.2})
    
    def test_default_values(self):
        """Test task default values."""
        task = Task(
            task_id="task_123",
            func_name="test_func"
        )
        
        self.assertEqual(task.args, ())
        self.assertEqual(task.kwargs, {})
        self.assertEqual(task.priority, 0)
        self.assertEqual(task.max_retries, 3)
        self.assertIsNone(task.timeout)


class TestLocalQueue(unittest.TestCase):
    """Tests for LocalQueue."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.queue = LocalQueue(max_workers=2)
    
    def tearDown(self):
        """Clean up after tests."""
        self.queue.shutdown(wait=True)
    
    def test_create_queue(self):
        """Test creating a local queue."""
        queue = LocalQueue(max_workers=4)
        self.assertIsNotNone(queue)
        self.assertEqual(queue.max_workers, 4)
        queue.shutdown()
    
    def test_register_function(self):
        """Test registering a custom function."""
        def custom_func(x):
            return x * 2
        
        self.queue.register_function('custom', custom_func)
        self.assertIn('custom', self.queue._functions)
    
    def test_submit_task(self):
        """Test submitting a task."""
        def test_func():
            return "success"
        
        self.queue.register_function('test_func', test_func)
        task_id = self.queue.submit_task('test_func')
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.queue.tasks)
    
    def test_get_result(self):
        """Test getting task result."""
        def add(a, b):
            return a + b
        
        self.queue.register_function('add', add)
        task_id = self.queue.submit_task('add', 2, 3)
        
        result = self.queue.get_result(task_id, timeout=5)
        
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, 5)
    
    def test_get_result_with_kwargs(self):
        """Test task execution with keyword arguments."""
        def multiply(a, b):
            return a * b
        
        self.queue.register_function('multiply', multiply)
        task_id = self.queue.submit_task('multiply', a=3, b=4)
        
        result = self.queue.get_result(task_id, timeout=5)
        
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, 12)
    
    def test_task_failure(self):
        """Test task failure handling."""
        def failing_func():
            raise ValueError("Test error")
        
        self.queue.register_function('failing_func', failing_func)
        task_id = self.queue.submit_task('failing_func', max_retries=0)
        
        result = self.queue.get_result(task_id, timeout=5)
        
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("Test error", result.error)
    
    def test_task_retry(self):
        """Test task retry logic."""
        call_count = [0]
        
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary error")
            return "success"
        
        self.queue.register_function('flaky_func', flaky_func)
        task_id = self.queue.submit_task('flaky_func', max_retries=3)
        
        result = self.queue.get_result(task_id, timeout=10)
        
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "success")
        self.assertGreater(result.retry_count, 0)
    
    def test_cancel_task(self):
        """Test cancelling a task."""
        def slow_func():
            time.sleep(10)
            return "done"
        
        self.queue.register_function('slow_func', slow_func)
        task_id = self.queue.submit_task('slow_func')
        
        # Try to cancel immediately
        cancelled = self.queue.cancel_task(task_id)
        
        # May or may not succeed depending on timing
        self.assertIsInstance(cancelled, bool)
    
    def test_get_queue_stats(self):
        """Test getting queue statistics."""
        def quick_func():
            return "done"
        
        self.queue.register_function('quick_func', quick_func)
        
        # Submit multiple tasks
        task_ids = [self.queue.submit_task('quick_func') for _ in range(3)]
        
        # Wait for completion
        for task_id in task_ids:
            self.queue.get_result(task_id, timeout=5)
        
        stats = self.queue.get_queue_stats()
        
        self.assertIn('total', stats)
        self.assertIn('completed', stats)
        self.assertEqual(stats['total'], 3)
    
    def test_gather_results(self):
        """Test gathering multiple results."""
        def square(x):
            return x ** 2
        
        self.queue.register_function('square', square)
        
        task_ids = [
            self.queue.submit_task('square', i)
            for i in range(5)
        ]
        
        results = self.queue.gather_results(task_ids, timeout=5)
        
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.status == TaskStatus.COMPLETED for r in results))
        self.assertEqual([r.result for r in results], [0, 1, 4, 9, 16])
    
    def test_gather_results_with_error(self):
        """Test gathering results with raise_on_error."""
        def conditional_func(x):
            if x == 2:
                raise ValueError("Error for 2")
            return x
        
        self.queue.register_function('conditional_func', conditional_func)
        
        task_ids = [
            self.queue.submit_task('conditional_func', i, max_retries=0)
            for i in range(4)
        ]
        
        with self.assertRaises(RuntimeError):
            self.queue.gather_results(task_ids, timeout=5, raise_on_error=True)
    
    def test_unknown_function(self):
        """Test submitting task with unknown function."""
        task_id = self.queue.submit_task('nonexistent_func', max_retries=0)
        result = self.queue.get_result(task_id, timeout=5)
        
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("Unknown function", result.error)
    
    def test_parallel_execution(self):
        """Test parallel task execution."""
        execution_times = []
        
        def timed_func(duration):
            start = time.time()
            time.sleep(duration)
            execution_times.append(time.time() - start)
            return duration
        
        self.queue.register_function('timed_func', timed_func)
        
        # Submit 4 tasks with 0.5s each to 2 workers
        # Should take ~1s total (2 parallel batches)
        start_time = time.time()
        task_ids = [
            self.queue.submit_task('timed_func', 0.5)
            for _ in range(4)
        ]
        
        results = self.queue.gather_results(task_ids, timeout=10)
        total_time = time.time() - start_time
        
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.status == TaskStatus.COMPLETED for r in results))
        # Should complete in ~1s (parallel), not 2s (sequential)
        self.assertLess(total_time, 1.5)


class TestDistributedBatchProcessor(unittest.TestCase):
    """Tests for DistributedBatchProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.queue = LocalQueue(max_workers=2)
        self.processor = DistributedBatchProcessor(self.queue)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        self.queue.shutdown(wait=True)
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
                completed_at=datetime.now() + timedelta(seconds=1)
            ),
            TaskResult(
                task_id="task_2",
                status=TaskStatus.COMPLETED,
                started_at=datetime.now(),
                completed_at=datetime.now() + timedelta(seconds=2)
            ),
            TaskResult(
                task_id="task_3",
                status=TaskStatus.FAILED,
                error="Test error"
            )
        ]
        
        stats = self.processor.aggregate_statistics(results)
        
        self.assertEqual(stats['total_tasks'], 3)
        self.assertEqual(stats['successful'], 2)
        self.assertEqual(stats['failed'], 1)
        self.assertGreater(stats['avg_duration'], 0)
        self.assertEqual(len(stats['errors']), 1)


class TestCreateQueue(unittest.TestCase):
    """Tests for create_queue factory function."""
    
    def test_create_local_queue(self):
        """Test creating local queue."""
        queue = create_queue('local', max_workers=4)
        
        self.assertIsInstance(queue, LocalQueue)
        self.assertEqual(queue.max_workers, 4)
        queue.shutdown()
    
    def test_create_unknown_backend(self):
        """Test error for unknown backend."""
        with self.assertRaises(ValueError):
            create_queue('nonexistent')
    
    def test_create_redis_queue_without_dependency(self):
        """Test redis queue without redis installed."""
        with self.assertRaises(ImportError):
            create_queue('redis')
    
    def test_create_celery_queue_without_dependency(self):
        """Test celery queue without celery installed."""
        with self.assertRaises(ImportError):
            create_queue('celery')


if __name__ == '__main__':
    unittest.main()
