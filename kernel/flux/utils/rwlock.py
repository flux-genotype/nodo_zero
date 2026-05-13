import threading

class RWLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    def reader(self):
        return _ReadLock(self)

    def writer(self):
        return _WriteLock(self)

class _ReadLock:
    def __init__(self, rwlock: RWLock):
        self.rwlock = rwlock

    def __enter__(self):
        with self.rwlock._read_ready:
            while self.rwlock._writer_active or self.rwlock._writers_waiting > 0:
                self.rwlock._read_ready.wait()
            self.rwlock._readers += 1

    def __exit__(self, *args):
        with self.rwlock._read_ready:
            self.rwlock._readers -= 1
            if self.rwlock._readers == 0:
                self.rwlock._read_ready.notify_all()

class _WriteLock:
    def __init__(self, rwlock: RWLock):
        self.rwlock = rwlock

    def __enter__(self):
        with self.rwlock._read_ready:
            self.rwlock._writers_waiting += 1
            while self.rwlock._readers > 0 or self.rwlock._writer_active:
                self.rwlock._read_ready.wait()
            self.rwlock._writers_waiting -= 1
            self.rwlock._writer_active = True

    def __exit__(self, *args):
        with self.rwlock._read_ready:
            self.rwlock._writer_active = False
            self.rwlock._read_ready.notify_all()
