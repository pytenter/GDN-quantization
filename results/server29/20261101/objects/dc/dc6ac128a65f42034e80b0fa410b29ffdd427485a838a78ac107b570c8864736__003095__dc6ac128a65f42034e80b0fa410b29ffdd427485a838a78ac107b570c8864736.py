def has_aliasing_or_mutation_cache_on_self(self):
            try:
                return self.__has_aliasing_or_mutation_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__has_aliasing_or_mutation_cache", rv)
            return rv
        