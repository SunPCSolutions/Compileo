"""
Retry utilities for extraction operations.

This module provides retry logic and fallback mechanisms for handling
transient failures in extraction workflows.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, Type, TypeVar, Union
from functools import wraps

from .exceptions import (
    APIConnectionError,
    APIRateLimitError,
    APIAuthenticationError,
    ClassificationFailureError,
    ExtractionError
)

T = TypeVar('T')

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple] = None
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Exponential backoff multiplier
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that should be retried
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

        if retryable_exceptions is None:
            self.retryable_exceptions = (
                APIConnectionError,
                APIRateLimitError,
                ConnectionError,
                TimeoutError,
                OSError
            )
        else:
            self.retryable_exceptions = retryable_exceptions


def is_retryable_exception(exception: Exception, retryable_exceptions: tuple) -> bool:
    """Check if an exception should trigger a retry."""
    return isinstance(exception, retryable_exceptions)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for the given attempt number."""
    delay = config.initial_delay * (config.backoff_factor ** (attempt - 1))
    delay = min(delay, config.max_delay)

    if config.jitter:
        import random
        # Add up to 25% jitter
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def retry_with_config(
    config: RetryConfig,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator that adds retry logic with custom configuration.

    Args:
        config: Retry configuration
        logger: Optional logger for retry attempts

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception = Exception("Unknown error")

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not is_retryable_exception(e, config.retryable_exceptions):
                        # Not a retryable exception, re-raise immediately
                        raise

                    if attempt == config.max_attempts:
                        # Last attempt failed, re-raise
                        if logger:
                            logger.error(
                                f"Operation failed after {config.max_attempts} attempts: {e}"
                            )
                        raise last_exception

                    # Calculate delay and wait
                    delay = calculate_delay(attempt, config)
                    if logger:
                        logger.warning(
                            f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s..."
                        )

                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


def retry_async_with_config(
    config: RetryConfig,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator that adds retry logic for async functions with custom configuration.

    Args:
        config: Retry configuration
        logger: Optional logger for retry attempts

    Returns:
        Decorated async function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Exception = Exception("Unknown async error")

            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    else:
                        return result  # type: ignore
                except Exception as e:
                    last_exception = e

                    if not is_retryable_exception(e, config.retryable_exceptions):
                        # Not a retryable exception, re-raise immediately
                        raise

                    if attempt == config.max_attempts:
                        # Last attempt failed, re-raise
                        if logger:
                            logger.error(
                                f"Async operation failed after {config.max_attempts} attempts: {e}"
                            )
                        raise last_exception

                    # Calculate delay and wait
                    delay = calculate_delay(attempt, config)
                    if logger:
                        logger.warning(
                            f"Async attempt {attempt} failed: {e}. Retrying in {delay:.2f}s..."
                        )

                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper  # type: ignore
    return decorator


# Pre-configured retry decorators for common use cases
def retry_api_call(max_attempts: int = 3, logger: Optional[logging.Logger] = None):
    """Decorator for retrying API calls with appropriate configuration."""
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        retryable_exceptions=(APIConnectionError, APIRateLimitError, ConnectionError, TimeoutError)
    )
    return retry_with_config(config, logger)


def retry_classification(max_attempts: int = 2, logger: Optional[logging.Logger] = None):
    """Decorator for retrying classification operations."""
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=0.5,
        max_delay=10.0,
        backoff_factor=1.5,
        retryable_exceptions=(ClassificationFailureError, APIConnectionError)
    )
    return retry_with_config(config, logger)


def retry_storage_operation(max_attempts: int = 3, logger: Optional[logging.Logger] = None):
    """Decorator for retrying storage operations."""
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=0.1,
        max_delay=5.0,
        backoff_factor=2.0,
        retryable_exceptions=(ConnectionError, OSError, TimeoutError)
    )
    return retry_with_config(config, logger)


class FallbackChain:
    """
    Chain of fallback operations to try in sequence.

    Useful for providing alternative approaches when primary operations fail.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.fallbacks = []
        self.logger = logger or logging.getLogger(__name__)

    def add_fallback(self, func: Callable, *args, **kwargs):
        """Add a fallback function with its arguments."""
        self.fallbacks.append((func, args, kwargs))
        return self

    def execute(self) -> Any:
        """
        Execute the fallback chain.

        Returns the result of the first successful operation,
        or raises the last exception if all fail.
        """
        last_exception: Exception = Exception("No fallback operations configured")

        for i, (func, args, kwargs) in enumerate(self.fallbacks):
            try:
                if self.logger:
                    self.logger.info(f"Trying fallback {i + 1}/{len(self.fallbacks)}")
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if self.logger:
                    self.logger.warning(f"Fallback {i + 1} failed: {e}")
                continue

        # All fallbacks failed
        if self.logger:
            self.logger.error("All fallback operations failed")
        raise last_exception


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for protecting against cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds to wait before trying to close circuit
            expected_exception: Type of exception to count as failure
            logger: Optional logger
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.logger = logger or logging.getLogger(__name__)

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.state != 'open':
            return True

        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == 'open':
            if not self._should_attempt_reset():
                raise Exception("Circuit breaker is open")

            # Try to close the circuit
            self.state = 'half_open'
            if self.logger:
                self.logger.info("Circuit breaker attempting to close")

        try:
            result = func(*args, **kwargs)

            # Success - reset failure count and close circuit
            if self.state == 'half_open':
                if self.logger:
                    self.logger.info("Circuit breaker successfully closed")
                self.state = 'closed'

            self.failure_count = 0
            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                if self.logger:
                    self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            raise

        except Exception as e:
            # Non-expected exception, don't count as circuit failure
            raise