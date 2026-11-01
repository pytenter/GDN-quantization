def _get_atomic_add_buffers_cache_on_self(self):
            try:
                return self.___get_atomic_add_buffers_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "___get_atomic_add_buffers_cache", rv)
            return rv
        