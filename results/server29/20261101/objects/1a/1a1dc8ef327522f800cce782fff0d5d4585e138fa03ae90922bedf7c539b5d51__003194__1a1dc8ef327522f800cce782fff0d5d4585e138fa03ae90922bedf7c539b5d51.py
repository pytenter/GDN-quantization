def inplace_update_buffers_cache_on_self(self):
            try:
                return self.__inplace_update_buffers_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__inplace_update_buffers_cache", rv)
            return rv
        