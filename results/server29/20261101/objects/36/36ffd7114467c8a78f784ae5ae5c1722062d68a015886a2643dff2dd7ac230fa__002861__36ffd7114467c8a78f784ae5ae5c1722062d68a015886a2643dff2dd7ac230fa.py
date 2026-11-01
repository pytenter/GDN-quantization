def inner_fn_str_cache_on_self(self):
            try:
                return self.__inner_fn_str_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__inner_fn_str_cache", rv)
            return rv
        