def get_name_cache_on_self(self):
            try:
                return self.__get_name_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_name_cache", rv)
            return rv
        