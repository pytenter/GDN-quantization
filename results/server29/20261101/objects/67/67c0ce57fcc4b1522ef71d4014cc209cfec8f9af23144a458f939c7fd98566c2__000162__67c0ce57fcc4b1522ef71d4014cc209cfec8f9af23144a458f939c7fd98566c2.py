def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('pre_self_positional','self_arg','post_self_positional','pre_tensor_options_kwarg_only','tensor_options','post_tensor_options_kwarg_only','out',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__