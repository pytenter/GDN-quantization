def op_counts_cache_on_self(self):
            try:
                return self.__op_counts_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__op_counts_cache", rv)
            return rv
        