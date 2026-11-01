def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('device','dtype','inner_fn','ranges','scan_ranges','size','combine_fn','reindex','reduction_hint','output_index','dtypes','inner_fns',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__