def is_reduction_cache_on_self(self):
            try:
                return self.__is_reduction_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__is_reduction_cache", rv)
            return rv
        