def scheduler_nodes_cache_on_self(self):
            try:
                return self.__scheduler_nodes_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__scheduler_nodes_cache", rv)
            return rv
        