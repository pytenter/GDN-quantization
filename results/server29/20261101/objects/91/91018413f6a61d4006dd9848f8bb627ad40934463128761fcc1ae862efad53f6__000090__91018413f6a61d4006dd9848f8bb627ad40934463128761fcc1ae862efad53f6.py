def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('dispatch_key','use_out_as_primary','device_guard','external','index',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__