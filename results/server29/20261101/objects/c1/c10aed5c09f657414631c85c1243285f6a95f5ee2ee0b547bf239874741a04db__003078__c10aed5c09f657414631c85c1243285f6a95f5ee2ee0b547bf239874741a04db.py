def can_codegen_in_low_precision_cache_on_self(self):
            try:
                return self.__can_codegen_in_low_precision_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__can_codegen_in_low_precision_cache", rv)
            return rv
        