def buffer_read_counts_cache_on_self(self):
            try:
                return self.__buffer_read_counts_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__buffer_read_counts_cache", rv)
            return rv
        