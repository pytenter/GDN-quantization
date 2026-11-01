def is_split_scan_cache_on_self(self):
            try:
                return self.__is_split_scan_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__is_split_scan_cache", rv)
            return rv
        