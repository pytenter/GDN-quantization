def num_reduction_dims_cache_on_self(self):
            try:
                return self.__num_reduction_dims_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__num_reduction_dims_cache", rv)
            return rv
        