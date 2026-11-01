def get_default_sizes_body_cache_on_self(self):
            try:
                return self.__get_default_sizes_body_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_default_sizes_body_cache", rv)
            return rv
        