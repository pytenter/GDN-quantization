def bounds_cache_on_self(self):
            try:
                return self.__bounds_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__bounds_cache", rv)
            return rv
        