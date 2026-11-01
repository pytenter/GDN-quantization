def is_template_cache_on_self(self):
            try:
                return self.__is_template_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__is_template_cache", rv)
            return rv
        