def symt_cache_on_self(self):
            try:
                return self.__symt_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__symt_cache", rv)
            return rv
        