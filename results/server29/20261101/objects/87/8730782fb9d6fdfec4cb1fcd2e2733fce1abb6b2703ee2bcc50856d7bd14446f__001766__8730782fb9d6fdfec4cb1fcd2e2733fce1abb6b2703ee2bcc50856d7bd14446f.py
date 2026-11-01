def __create_fn__(cls, FrozenInstanceError):
 def __delattr__(self,name):
  if type(self) is cls or name in ('reduce_op','mask_buffer','offset_shape','offset_dim',):
   raise FrozenInstanceError(f"cannot delete field {name!r}")
  super(cls, self).__delattr__(name)
 return __delattr__