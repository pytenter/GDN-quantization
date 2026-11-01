def buf_accesses_cache_on_self(self):
            try:
                return self.__buf_accesses_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__buf_accesses_cache", rv)
            return rv
        