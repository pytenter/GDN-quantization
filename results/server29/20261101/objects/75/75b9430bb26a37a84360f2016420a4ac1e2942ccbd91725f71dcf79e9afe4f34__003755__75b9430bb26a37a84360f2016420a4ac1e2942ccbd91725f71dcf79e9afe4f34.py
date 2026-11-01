def __create_fn__(cls, FrozenInstanceError):
 def __setattr__(self,name,value):
  if type(self) is cls or name in ('path_in_repo','file_path','lock_path','metadata_path',):
   raise FrozenInstanceError(f"cannot assign to field {name!r}")
  super(cls, self).__setattr__(name, value)
 return __setattr__