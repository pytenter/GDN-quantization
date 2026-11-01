def can_codegen_without_upcasts_cache_on_self(self):
            try:
                return self.__can_codegen_without_upcasts_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__can_codegen_without_upcasts_cache", rv)
            return rv
        