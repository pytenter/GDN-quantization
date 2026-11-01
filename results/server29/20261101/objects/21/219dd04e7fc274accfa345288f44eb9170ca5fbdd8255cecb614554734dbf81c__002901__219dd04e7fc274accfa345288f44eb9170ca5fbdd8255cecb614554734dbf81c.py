def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('device','dtype','inner_fn','ranges','sort_ranges','size','reindex','reduction_hint','output_index','dtypes','inner_fns','stable','descending',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__