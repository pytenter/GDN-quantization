def get_estimated_runtime_cache_on_self(self):
            try:
                return self.__get_estimated_runtime_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_estimated_runtime_cache", rv)
            return rv
        