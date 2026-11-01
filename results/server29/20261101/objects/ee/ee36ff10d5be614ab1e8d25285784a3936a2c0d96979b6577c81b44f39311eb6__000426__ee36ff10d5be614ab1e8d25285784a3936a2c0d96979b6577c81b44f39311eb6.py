def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('frame_id','frame_compile_id','compiled_autograd_id',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__