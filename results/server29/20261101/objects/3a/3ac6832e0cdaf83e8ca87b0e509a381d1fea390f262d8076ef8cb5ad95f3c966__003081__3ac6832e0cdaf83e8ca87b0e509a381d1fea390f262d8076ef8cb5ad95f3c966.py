def get_read_buffer_sizes_cache_on_self(self):
            try:
                return self.__get_read_buffer_sizes_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_read_buffer_sizes_cache", rv)
            return rv
        