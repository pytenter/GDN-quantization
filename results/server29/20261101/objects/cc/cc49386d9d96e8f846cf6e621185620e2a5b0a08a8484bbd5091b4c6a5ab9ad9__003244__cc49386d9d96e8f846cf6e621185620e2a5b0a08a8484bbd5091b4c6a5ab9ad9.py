def get_output_refs_cache_on_self(self):
            try:
                return self.__get_output_refs_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_output_refs_cache", rv)
            return rv
        