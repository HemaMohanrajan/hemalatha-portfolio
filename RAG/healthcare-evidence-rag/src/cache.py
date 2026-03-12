import time

class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 900):
        self.ttl = ttl_seconds
        self.store = {}  # key -> (expires_at, value)

    def get(self, key: str):
        item = self.store.get(key)
        if not item:
            return None
        exp, val = item
        if time.time() > exp:
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, value):
        self.store[key] = (time.time() + self.ttl, value)