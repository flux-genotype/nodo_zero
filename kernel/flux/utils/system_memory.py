import logging
import psutil

logger = logging.getLogger(__name__)

class SystemMemoryManager:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def check_memory(self) -> bool:
        mem = psutil.virtual_memory()
        if mem.percent / 100 > self.threshold:
            logger.warning("High system memory usage: %s%%", mem.percent)
            return False
        return True
