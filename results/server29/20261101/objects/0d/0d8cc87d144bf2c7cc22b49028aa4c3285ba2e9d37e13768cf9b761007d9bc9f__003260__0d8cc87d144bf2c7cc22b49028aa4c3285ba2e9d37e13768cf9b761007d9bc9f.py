def _get_reduction_index_coeffs_cache_on_self(self):
            try:
                return self.___get_reduction_index_coeffs_cache
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "___get_reduction_index_coeffs_cache", rv)
            return rv
        