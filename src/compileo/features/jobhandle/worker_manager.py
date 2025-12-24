"""
Worker management and scaling utilities for the enhanced job queue system.
Provides automated worker scaling, health monitoring, and load balancing.
"""

import os
import time
import signal
import psutil
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import subprocess

from .enhanced_job_queue import enhanced_job_queue_manager, perform_health_check
from .models import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for worker processes."""
    worker_name: str
    queue_name: str = "extraction_jobs"
    redis_url: str = "redis://localhost:6379/0"
    max_jobs: Optional[int] = None
    worker_timeout: int = 3600  # 1 hour
    burst: bool = False
    logging_level: str = "INFO"
    sentry_dsn: Optional[str] = None


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    min_workers: int = 1
    max_workers: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 80.0  # Scale up when CPU/memory > 80%
    scale_down_threshold: float = 30.0  # Scale down when CPU/memory < 30%
    cooldown_period: int = 300  # 5 minutes between scaling actions
    evaluation_period: int = 60  # Evaluate every 60 seconds


@dataclass
class WorkerProcess:
    """Represents a worker process."""
    pid: int
    worker_name: str
    start_time: datetime
    config: WorkerConfig
    status: str = "running"  # running, stopped, failed
    jobs_processed: int = 0
    last_heartbeat: Optional[datetime] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0


class WorkerManager:
    """Manages worker processes with auto-scaling capabilities."""

    def __init__(self, base_config: WorkerConfig, scaling_config: Optional[ScalingConfig] = None):
        self.base_config = base_config
        self.scaling_config = scaling_config or ScalingConfig()
        self.workers: Dict[int, WorkerProcess] = {}
        self._running = False
        self._scaling_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self.last_scale_action = datetime.min
        self.scale_cooldown_active = False

    def start_worker(self, worker_name: Optional[str] = None) -> Optional[int]:
        """Start a new worker process."""
        config = self.base_config
        if worker_name:
            config = WorkerConfig(**self.base_config.__dict__)
            config.worker_name = worker_name

        try:
            # Start worker process
            cmd = self._build_worker_command(config)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            worker = WorkerProcess(
                pid=process.pid,
                worker_name=config.worker_name,
                start_time=datetime.utcnow(),
                config=config
            )

            self.workers[process.pid] = worker
            logger.info(f"Started worker {config.worker_name} with PID {process.pid}")

            return process.pid

        except Exception as e:
            logger.error(f"Failed to start worker {worker_name}: {e}")
            return None

    def stop_worker(self, pid: int, graceful: bool = True) -> bool:
        """Stop a worker process."""
        if pid not in self.workers:
            return False

        worker = self.workers[pid]

        try:
            if graceful:
                # Send SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)

                # Wait up to 30 seconds for graceful shutdown
                for _ in range(30):
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(1)

            if psutil.pid_exists(pid):
                # Force kill if still running
                os.kill(pid, signal.SIGKILL)
                logger.warning(f"Force killed worker {worker.worker_name} (PID {pid})")

            worker.status = "stopped"
            logger.info(f"Stopped worker {worker.worker_name} (PID {pid})")
            return True

        except ProcessLookupError:
            # Process already dead
            worker.status = "stopped"
            return True
        except Exception as e:
            logger.error(f"Failed to stop worker {worker.worker_name}: {e}")
            worker.status = "failed"
            return False

    def stop_all_workers(self, graceful: bool = True) -> int:
        """Stop all worker processes."""
        stopped_count = 0
        for pid in list(self.workers.keys()):
            if self.stop_worker(pid, graceful):
                stopped_count += 1

        logger.info(f"Stopped {stopped_count} workers")
        return stopped_count

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about all workers."""
        total_workers = len(self.workers)
        running_workers = len([w for w in self.workers.values() if w.status == "running"])
        stopped_workers = len([w for w in self.workers.values() if w.status == "stopped"])
        failed_workers = len([w for w in self.workers.values() if w.status == "failed"])

        total_jobs_processed = sum(w.jobs_processed for w in self.workers.values())

        return {
            'total_workers': total_workers,
            'running_workers': running_workers,
            'stopped_workers': stopped_workers,
            'failed_workers': failed_workers,
            'total_jobs_processed': total_jobs_processed,
            'workers': [
                {
                    'pid': w.pid,
                    'name': w.worker_name,
                    'status': w.status,
                    'start_time': w.start_time.isoformat(),
                    'jobs_processed': w.jobs_processed,
                    'cpu_percent': w.cpu_percent,
                    'memory_mb': w.memory_mb,
                    'uptime_seconds': (datetime.utcnow() - w.start_time).total_seconds()
                }
                for w in self.workers.values()
            ]
        }

    def start_auto_scaling(self) -> None:
        """Start automatic scaling based on system load."""
        if not self.scaling_config or self._scaling_thread:
            return

        self._running = True
        self._scaling_thread = threading.Thread(target=self._auto_scaling_loop, daemon=True)
        self._scaling_thread.start()

        self._monitor_thread = threading.Thread(target=self._worker_monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Auto-scaling started")

    def stop_auto_scaling(self) -> None:
        """Stop automatic scaling."""
        self._running = False
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Auto-scaling stopped")

    def _auto_scaling_loop(self) -> None:
        """Main auto-scaling loop."""
        while self._running:
            try:
                self._evaluate_scaling()
                time.sleep(self.scaling_config.evaluation_period)
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                time.sleep(30)

    def _evaluate_scaling(self) -> None:
        """Evaluate system load and scale workers accordingly."""
        if self.scale_cooldown_active:
            if datetime.utcnow() - self.last_scale_action < timedelta(seconds=self.scaling_config.cooldown_period):
                return  # Still in cooldown
            else:
                self.scale_cooldown_active = False

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=5)
        memory_percent = psutil.virtual_memory().percent

        # Get queue stats
        queue_stats = enhanced_job_queue_manager.get_queue_stats()
        pending_jobs = queue_stats.get('pending_jobs', 0)
        running_jobs = queue_stats.get('running_jobs', 0)

        # Determine scaling action
        scale_up = (
            cpu_percent > self.scaling_config.scale_up_threshold or
            memory_percent > self.scaling_config.scale_up_threshold or
            pending_jobs > len(self.workers) * 5  # More than 5 pending jobs per worker
        )

        scale_down = (
            cpu_percent < self.scaling_config.scale_down_threshold and
            memory_percent < self.scaling_config.scale_down_threshold and
            pending_jobs == 0 and
            running_jobs < len(self.workers)  # Fewer running jobs than workers
        )

        current_workers = len([w for w in self.workers.values() if w.status == "running"])

        if scale_up and current_workers < self.scaling_config.max_workers:
            self._scale_up()
        elif scale_down and current_workers > self.scaling_config.min_workers:
            self._scale_down()

    def _scale_up(self) -> None:
        """Scale up by starting a new worker."""
        worker_name = f"auto_worker_{int(time.time())}"
        if self.start_worker(worker_name):
            self.last_scale_action = datetime.utcnow()
            self.scale_cooldown_active = True
            logger.info(f"Scaled up: started worker {worker_name}")

    def _scale_down(self) -> None:
        """Scale down by stopping an idle worker."""
        # Find the oldest idle worker
        idle_workers = [
            w for w in self.workers.values()
            if w.status == "running" and w.jobs_processed == 0
        ]

        if idle_workers:
            oldest_worker = min(idle_workers, key=lambda w: w.start_time)
            if self.stop_worker(oldest_worker.pid, graceful=True):
                self.last_scale_action = datetime.utcnow()
                self.scale_cooldown_active = True
                logger.info(f"Scaled down: stopped worker {oldest_worker.worker_name}")

    def _worker_monitor_loop(self) -> None:
        """Monitor worker health and update statistics."""
        while self._running:
            try:
                self._update_worker_stats()
                self._check_worker_health()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Worker monitor error: {e}")
                time.sleep(60)

    def _update_worker_stats(self) -> None:
        """Update statistics for all workers."""
        for worker in self.workers.values():
            if worker.status == "running" and psutil.pid_exists(worker.pid):
                try:
                    process = psutil.Process(worker.pid)
                    worker.cpu_percent = process.cpu_percent()
                    worker.memory_mb = process.memory_info().rss / (1024 * 1024)
                    worker.last_heartbeat = datetime.utcnow()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    worker.status = "failed"

    def _check_worker_health(self) -> None:
        """Check worker health and restart failed workers."""
        for worker in self.workers.values():
            if worker.status == "running":
                # Check if process is still alive
                if not psutil.pid_exists(worker.pid):
                    logger.warning(f"Worker {worker.worker_name} (PID {worker.pid}) died unexpectedly")
                    worker.status = "failed"
                    continue

                # Check for stale workers (no heartbeat for 5 minutes)
                if worker.last_heartbeat and (datetime.utcnow() - worker.last_heartbeat) > timedelta(minutes=5):
                    logger.warning(f"Worker {worker.worker_name} appears stale, restarting")
                    self.stop_worker(worker.pid, graceful=False)
                    # Restart will be handled by auto-scaling

    def _build_worker_command(self, config: WorkerConfig) -> List[str]:
        """Build the command to start a worker process."""
        cmd = [
            "python", "-m", "compileo.features.extraction.worker_manager",
            "start-worker",
            "--name", config.worker_name,
            "--queue", config.queue_name,
            "--redis", config.redis_url,
            "--log-level", config.logging_level
        ]

        if config.max_jobs:
            cmd.extend(["--max-jobs", str(config.max_jobs)])

        if config.worker_timeout:
            cmd.extend(["--timeout", str(config.worker_timeout)])

        if config.burst:
            cmd.append("--burst")

        if config.sentry_dsn:
            cmd.extend(["--sentry-dsn", config.sentry_dsn])

        return cmd


class WorkerPool:
    """Manages a pool of worker managers for different queues."""

    def __init__(self):
        self.managers: Dict[str, WorkerManager] = {}
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

    def add_manager(self, queue_name: str, manager: WorkerManager) -> None:
        """Add a worker manager for a specific queue."""
        self.managers[queue_name] = manager

    def start_all(self) -> None:
        """Start all worker managers."""
        for manager in self.managers.values():
            manager.start_auto_scaling()

        self._running = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

        logger.info(f"Started worker pool with {len(self.managers)} managers")

    def stop_all(self) -> None:
        """Stop all worker managers."""
        self._running = False
        for manager in self.managers.values():
            manager.stop_auto_scaling()

        if self._health_check_thread:
            self._health_check_thread.join(timeout=10)

        logger.info("Stopped worker pool")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire worker pool."""
        total_stats = {
            'total_managers': len(self.managers),
            'total_workers': 0,
            'running_workers': 0,
            'managers': {}
        }

        for queue_name, manager in self.managers.items():
            stats = manager.get_worker_stats()
            total_stats['managers'][queue_name] = stats
            total_stats['total_workers'] += stats['total_workers']
            total_stats['running_workers'] += stats['running_workers']

        return total_stats

    def _health_check_loop(self) -> None:
        """Periodic health check for the worker pool."""
        while self._running:
            try:
                # Perform health check
                health = perform_health_check()

                if health['status'] in ['warning', 'critical']:
                    logger.warning(f"System health check failed: {health['status']}")
                    # Could implement alerting here

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(60)


# Global worker pool
worker_pool = WorkerPool()


def create_worker_manager(
    worker_name_prefix: str = "extraction_worker",
    queue_name: str = "extraction_jobs",
    redis_url: str = "redis://localhost:6379/0",
    scaling_config: Optional[ScalingConfig] = None
) -> WorkerManager:
    """Create a configured worker manager."""
    base_config = WorkerConfig(
        worker_name=f"{worker_name_prefix}_{int(time.time())}",
        queue_name=queue_name,
        redis_url=redis_url
    )

    return WorkerManager(base_config, scaling_config)


def start_worker_process(
    name: str,
    queue: str = "extraction_jobs",
    redis_url: str = "redis://localhost:6379/0",
    max_jobs: Optional[int] = None,
    timeout: int = 3600,
    burst: bool = False,
    log_level: str = "INFO"
) -> None:
    """Start a worker process (called from command line)."""
    from .enhanced_job_queue import start_enhanced_worker

    logger.info(f"Starting worker {name} for queue {queue}")

    try:
        start_enhanced_worker(
            redis_url=redis_url,
            queue_name=queue,
            worker_name=name
        )
    except KeyboardInterrupt:
        logger.info(f"Worker {name} stopped by user")
    except Exception as e:
        logger.error(f"Worker {name} failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Worker Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start worker command
    start_parser = subparsers.add_parser("start-worker", help="Start a worker process")
    start_parser.add_argument("--name", required=True, help="Worker name")
    start_parser.add_argument("--queue", default="extraction_jobs", help="Queue name")
    start_parser.add_argument("--redis", default="redis://localhost:6379/0", help="Redis URL")
    start_parser.add_argument("--max-jobs", type=int, help="Maximum jobs per worker")
    start_parser.add_argument("--timeout", type=int, default=3600, help="Worker timeout")
    start_parser.add_argument("--burst", action="store_true", help="Burst mode")
    start_parser.add_argument("--log-level", default="INFO", help="Logging level")
    start_parser.add_argument("--sentry-dsn", help="Sentry DSN for error reporting")

    args = parser.parse_args()

    if args.command == "start-worker":
        start_worker_process(
            name=args.name,
            queue=args.queue,
            redis_url=args.redis,
            max_jobs=args.max_jobs,
            timeout=args.timeout,
            burst=args.burst,
            log_level=args.log_level
        )
    else:
        parser.print_help()