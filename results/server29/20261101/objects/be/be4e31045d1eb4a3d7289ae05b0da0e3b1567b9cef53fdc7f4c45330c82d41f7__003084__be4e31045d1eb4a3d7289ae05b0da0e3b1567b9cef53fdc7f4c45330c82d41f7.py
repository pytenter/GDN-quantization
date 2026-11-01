def pointwise_read_writes_cache_on_self(self):
            try:
                return self.__pointwise_read_writes_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__pointwise_read_writes_cache", rv)
            return rv
        