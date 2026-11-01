def max_parallel_depth_cache_on_self(self):
            try:
                return self.__max_parallel_depth_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__max_parallel_depth_cache", rv)
            return rv
        