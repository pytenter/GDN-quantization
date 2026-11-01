def get_template_node_cache_on_self(self):
            try:
                return self.__get_template_node_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "__get_template_node_cache", rv)
            return rv
        