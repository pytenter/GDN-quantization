def select_index_dtype_cache_on_self(self):
            try:
                return self.__select_index_dtype_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__select_index_dtype_cache", rv)
            return rv
        