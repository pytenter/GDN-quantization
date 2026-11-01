def get_nodes_cache_on_self(self):
            try:
                return self.__get_nodes_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_nodes_cache", rv)
            return rv
        