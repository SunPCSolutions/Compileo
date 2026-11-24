"""
Concurrent Worker for Compileo - Enables vertical scaling via ThreadPoolExecutor.

This module provides concurrent job processing within a single worker container
using Python's ThreadPoolExecutor. Each job runs in its own thread with proper
isolation and thread-safety measures.

Key features:
- Configurable concurrency level via WORKER_CONCURRENCY environment variable
- Thread-safe job execution with proper resource isolation
- Graceful shutdown with proper thread cleanup
- Compatible with existing RQ job queue system
- Automatic retry and error handling

Usage:
    Set WORKER_CONCURRENCY environment variable to control concurrency level.
    Default is 2. Set to 1 for sequential processing (backwards compatible).
    
    Total capacity = WORKER_REPLICAS (containers) × WORKER_CONCURRENCY (threads)
"""

import os
import sys
import time
import logging
import threading
import signal
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


class ThreadLocalResources:
    """
    Thread-local resource manager for database connections and other
    non-thread-safe resources.
    
    Each worker thread gets its own isolated resources to prevent
    conflicts and ensure thread safety.
    """
    
    def __init__(self):
        self._local = threading.local()
    
    def get_db_connection(self):
        """Get or create thread-local database connection."""
        if not hasattr(self._local, 'db_connection') or self._local.db_connection is None:
            from src.compileo.storage.src.database import get_db_connection
            self._local.db_connection = get_db_connection()
            logger.debug(f"Thread {threading.get_ident()}: Created new DB connection")
        return self._local.db_connection
    
    def close_db_connection(self):
        """Close thread-local database connection."""
        if hasattr(self._local, 'db_connection') and self._local.db_connection:
            try:
                self._local.db_connection.close()
                logger.debug(f"Thread {threading.get_ident()}: Closed DB connection")
            except Exception as e:
                logger.warning(f"Thread {threading.get_ident()}: Error closing DB connection: {e}")
            finally:
                self._local.db_connection = None


class ConcurrentWorker:
    """
    Concurrent job worker using ThreadPoolExecutor for vertical scaling.
    
    This worker processes multiple RQ jobs simultaneously within a single
    container, enabling more efficient resource utilization and higher
    throughput for I/O-bound workloads.
    
    Attributes:
        redis_url: Redis connection URL
        queues: List of queue names to listen to
        concurrency: Number of concurrent worker threads
        name: Worker name for identification
        default_worker_ttl: Maximum worker lifetime in seconds
    """
    
    def __init__(
        self,
        redis_url: str,
        queues: List[str],
        concurrency: int = 2,
        name: Optional[str] = None,
        default_worker_ttl: int = 270000
    ):
        """
        Initialize the ConcurrentWorker.
        
        Args:
            redis_url: Redis connection URL
            queues: List of RQ queue names to listen to
            concurrency: Number of concurrent worker threads (default: 2)
            name: Worker name (auto-generated if not provided)
            default_worker_ttl: Maximum worker lifetime before restart
        """
        self.redis_url = redis_url
        self.queues = queues
        self.concurrency = max(1, concurrency)
        self.name = name or f"concurrent_worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.default_worker_ttl = default_worker_ttl
        
        self._shutdown_event = threading.Event()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._active_jobs: Dict[int, str] = {}  # thread_id -> job_id
        self._lock = threading.Lock()
        self._thread_local = ThreadLocalResources()
        
        # Import Redis and RQ here to avoid import issues during module load
        from redis import Redis
        from rq import Queue
        
        self.redis_conn = Redis.from_url(redis_url)
        self.rq_queues = [
            Queue(name, connection=self.redis_conn)
            for name in queues
        ]
        
        logger.info(
            f"ConcurrentWorker '{self.name}' initialized: "
            f"concurrency={self.concurrency}, queues={queues}"
        )
    
    def work(self, burst: bool = False) -> None:
        """
        Start processing jobs.
        
        Args:
            burst: If True, process all available jobs and exit.
                  If False, run continuously until shutdown signal.
        """
        logger.info(
            f"Starting ConcurrentWorker '{self.name}' with {self.concurrency} threads"
        )
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix=f"{self.name}_thread"
        )
        
        try:
            if burst:
                self._work_burst()
            else:
                self._work_continuous()
        except Exception as e:
            logger.error(f"Worker '{self.name}' error: {e}", exc_info=True)
        finally:
            self._cleanup()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Worker '{self.name}' received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Ignore SIGHUP to prevent shutdown on terminal disconnect
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    def _work_burst(self) -> None:
        """Process all available jobs and exit."""
        logger.info(f"Worker '{self.name}' running in burst mode")
        
        futures = []
        
        # Collect and submit all available jobs
        for queue in self.rq_queues:
            while not self._shutdown_event.is_set():
                job = self._dequeue_job(queue)
                if job is None:
                    break
                
                future = self._executor.submit(self._execute_job, job, queue.name)
                futures.append((future, job))
                
                # Limit concurrent jobs to prevent memory issues
                if len(futures) >= self.concurrency * 2:
                    # Wait for at least one job to complete
                    done, _ = self._wait_for_any(futures)
                    futures = [(f, j) for f, j in futures if not f.done()]
        
        # Wait for remaining jobs
        for future, job in futures:
            try:
                future.result(timeout=self.default_worker_ttl)
            except Exception as e:
                logger.error(f"Job {job.id if hasattr(job, 'id') else job} failed: {e}")
    
    def _work_continuous(self) -> None:
        """Continuously process jobs until shutdown."""
        logger.info(f"Worker '{self.name}' running in continuous mode")
        
        start_time = time.time()
        running_futures: Dict[Any, tuple] = {}  # future -> (job, queue_name)
        
        while not self._shutdown_event.is_set():
            try:
                # Check if worker has exceeded its TTL
                if time.time() - start_time > self.default_worker_ttl:
                    logger.info(f"Worker '{self.name}' reached TTL, initiating graceful shutdown...")
                    self._shutdown_event.set()
                    break
                
                # Clean up completed futures
                completed = [f for f in running_futures if f.done()]
                for future in completed:
                    job, queue_name = running_futures.pop(future)
                    try:
                        future.result()
                        logger.debug(f"Job {job.id} completed successfully")
                    except Exception as e:
                        logger.error(f"Job {job.id} failed: {e}")
                
                # Submit new jobs up to concurrency limit
                while (len(running_futures) < self.concurrency and 
                       not self._shutdown_event.is_set()):
                    
                    job, queue = self._dequeue_from_any_queue()
                    if job is None:
                        break
                    
                    future = self._executor.submit(self._execute_job, job, queue.name)
                    running_futures[future] = (job, queue.name)
                    logger.debug(f"Submitted job {job.id}, active jobs: {len(running_futures)}")
                
                # Brief pause to prevent tight polling loop
                if not running_futures:
                    time.sleep(0.1)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in work loop: {e}", exc_info=True)
                time.sleep(1)
        
        # Graceful shutdown: wait for running jobs
        if running_futures:
            logger.info(
                f"Waiting for {len(running_futures)} active jobs to complete..."
            )
            for future, (job, queue_name) in running_futures.items():
                try:
                    # Use shorter timeout for graceful shutdown
                    future.result(timeout=60)
                except Exception as e:
                    logger.error(f"Job {job.id} failed during shutdown: {e}")
    
    def _dequeue_from_any_queue(self):
        """
        Dequeue a job from any of the configured queues.
        
        Returns:
            Tuple of (job, queue) or (None, None)
        """
        for queue in self.rq_queues:
            job = self._dequeue_job(queue)
            if job:
                return job, queue
        return None, None
    
    def _dequeue_job(self, queue):
        """
        Dequeue a single job from the specified queue.
        
        Args:
            queue: RQ Queue instance
            
        Returns:
            RQ Job instance or None
        """
        try:
            # Use RQ's built-in pop mechanism
            job = queue.dequeue()
            if job:
                logger.debug(f"Dequeued job {job.id} from queue {queue.name}")
                return job
        except Exception as e:
            logger.error(f"Error dequeuing from queue {queue.name}: {e}")
        
        return None
    
    def _execute_job(self, job, queue_name: str) -> None:
        """
        Execute a single job in a worker thread.
        
        Args:
            job: RQ Job instance
            queue_name: Name of the queue the job came from
        """
        thread_id = threading.get_ident()
        job_id = job.id
        
        try:
            # Track active job
            with self._lock:
                self._active_jobs[thread_id] = job_id
            
            logger.debug(f"Thread {thread_id} starting job {job_id}")
            
            # Mark job as started in RQ
            job.started_at = datetime.utcnow()
            job.save(include_meta=False)
            
            # Execute the job using our worker wrapper
            # This ensures proper initialization and database connection handling
            self._run_job_wrapper(job_id)
            
            # Mark job as completed
            job.ended_at = datetime.utcnow()
            job.save(include_meta=False)
            
            # Clean up the job from the queue
            job.delete()
            
            logger.debug(f"Thread {thread_id} completed job {job_id}")
            
        except Exception as e:
            logger.error(f"Thread {thread_id} failed job {job_id}: {e}", exc_info=True)
            
            # Mark job as failed
            try:
                job.ended_at = datetime.utcnow()
                job.save(include_meta=False)
                # Re-queue failed job if it has retries left
                job.requeue()
            except Exception as requeue_error:
                logger.error(f"Failed to requeue job {job_id}: {requeue_error}")
        finally:
            # Clean up thread resources
            self._thread_local.close_db_connection()
            
            # Remove from active jobs
            with self._lock:
                self._active_jobs.pop(thread_id, None)
    
    def _run_job_wrapper(self, job_id: str) -> None:
        """
        Execute job using the worker wrapper.
        
        This method calls the same worker wrapper used by SimpleWorker,
        ensuring consistent job execution behavior.
        
        Args:
            job_id: ID of the job to execute
        """
        # Import here to ensure proper module path setup
        from src.compileo.features.jobhandle.worker_wrapper import process_job
        
        # Execute the job
        process_job(job_id)
    
    def _wait_for_any(self, futures: List[tuple], timeout: Optional[float] = None):
        """
        Wait for any of the given futures to complete.
        
        Args:
            futures: List of (future, job) tuples
            timeout: Maximum time to wait
            
        Returns:
            Tuple of (done, not_done) sets
        """
        from concurrent.futures import wait, FIRST_COMPLETED
        
        if not futures:
            return set(), set()
        
        future_list = [f for f, _ in futures]
        done, not_done = wait(
            future_list,
            timeout=timeout,
            return_when=FIRST_COMPLETED
        )
        return done, not_done
    
    def _cleanup(self) -> None:
        """Clean up worker resources on shutdown."""
        logger.info(f"Worker '{self.name}' cleaning up...")
        
        # Shutdown executor
        if self._executor:
            try:
                self._executor.shutdown(wait=True, timeout=60)
                logger.info(f"Worker '{self.name}' executor shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down executor: {e}")
            finally:
                self._executor = None
        
        logger.info(f"Worker '{self.name}' stopped")
    
    def get_active_job_count(self) -> int:
        """Get number of currently executing jobs."""
        with self._lock:
            return len(self._active_jobs)
    
    def get_active_job_ids(self) -> List[str]:
        """Get IDs of currently executing jobs."""
        with self._lock:
            return list(self._active_jobs.values())


def create_worker(
    queues: List[str],
    redis_url: str,
    concurrency: int = 2,
    worker_name: Optional[str] = None,
    default_worker_ttl: int = 270000
) -> ConcurrentWorker:
    """
    Factory function to create a ConcurrentWorker instance.
    
    Args:
        queues: List of queue names to listen to
        redis_url: Redis connection URL
        concurrency: Number of concurrent worker threads
        worker_name: Optional worker name
        default_worker_ttl: Worker timeout in seconds
        
    Returns:
        Configured ConcurrentWorker instance
    """
    return ConcurrentWorker(
        redis_url=redis_url,
        queues=queues,
        concurrency=concurrency,
        name=worker_name,
        default_worker_ttl=default_worker_ttl
    )
