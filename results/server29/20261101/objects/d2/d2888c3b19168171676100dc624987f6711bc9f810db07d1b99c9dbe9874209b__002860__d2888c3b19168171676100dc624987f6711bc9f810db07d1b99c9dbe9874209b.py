def inner_fn_opcount_cache_on_self(self):
            try:
                return self.__inner_fn_opcount_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__inner_fn_opcount_cache", rv)
            return rv
        