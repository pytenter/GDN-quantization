def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('height','width','longest_edge','shortest_edge','max_height','max_width',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__