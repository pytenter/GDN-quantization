def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('device','dtype','inner_fn','ranges','scan_ranges','size','combine_fn','reindex','reduction_hint','output_index','dtypes','inner_fns',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__