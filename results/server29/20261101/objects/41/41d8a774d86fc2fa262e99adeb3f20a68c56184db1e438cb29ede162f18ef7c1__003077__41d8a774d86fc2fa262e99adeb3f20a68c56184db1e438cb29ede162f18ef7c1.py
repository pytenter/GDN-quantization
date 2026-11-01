def get_buffer_names_cache_on_self(self):
            try:
                return self.__get_buffer_names_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_buffer_names_cache", rv)
            return rv
        