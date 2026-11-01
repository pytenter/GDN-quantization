def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('local_name','is_input','dynamism','is_derefed_cell_contents',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__