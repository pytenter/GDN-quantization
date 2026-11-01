def get_reduction_prefixes_cache_on_self(self):
            try:
                return self.__get_reduction_prefixes_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_reduction_prefixes_cache", rv)
            return rv
        