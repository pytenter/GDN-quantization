def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('name','args','kwargs','_param_ids_from','_param_ids_generated',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__