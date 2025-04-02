import os
import gc
import logging
import tracemalloc
import numpy as np
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_memory_usage(location):
    process_memory = 0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024 / 1024
        logger.warning(f"MEMORY USAGE at {location}: {process_memory:.2f} MB")
    except ImportError:
        logger.warning(f"MEMORY TRACKING at {location}: psutil not available")
    return process_memory

def log_array_info(name, arr):
    if isinstance(arr, np.ndarray):
        mem_usage = arr.nbytes / (1024 * 1024)
        logger.warning(f"ARRAY INFO - {name}: shape={arr.shape}, dtype={arr.dtype}, memory={mem_usage:.2f} MB")
    elif isinstance(arr, list):
        logger.warning(f"LIST INFO - {name}: length={len(arr)}")

class MemoryTracker:
    def __init__(self, location):
        self.location = location
        
    def __enter__(self):
        tracemalloc.start()
        log_memory_usage(f"start_{self.location}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        logger.warning(f"MEMORY STATS {self.location}: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
        tracemalloc.stop()
        log_memory_usage(f"end_{self.location}")
        gc.collect()
        log_memory_usage(f"after_gc_{self.location}")

def chunk_generator(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class WatchdogTimer:
    """A simple watchdog timer to detect when operations hang."""
    
    def __init__(self, timeout=60, operation_name="Operation"):
        self.timeout = timeout
        self.operation_name = operation_name
        self.timer = None
        self.start_time = None
        
    def _timeout_handler(self):
        elapsed = time.time() - self.start_time
        logger.warning(f"WATCHDOG ALERT: {self.operation_name} has been running for {elapsed:.1f} seconds, which exceeds timeout of {self.timeout} seconds")
        
    def start(self):
        """Start the watchdog timer."""
        self.start_time = time.time()
        self.timer = threading.Timer(self.timeout, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        
    def stop(self):
        """Stop the watchdog timer."""
        if self.timer:
            self.timer.cancel()
            self.timer = None