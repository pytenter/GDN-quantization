def write_triton_header_once_cache_on_self(self):
            try:
                return self.__write_triton_header_once_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__write_triton_header_once_cache", rv)
            return rv
        