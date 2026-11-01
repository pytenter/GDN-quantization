def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.file_path,self.lock_path,self.metadata_path,)==(other.file_path,other.lock_path,other.metadata_path,)
  return NotImplemented
 return __eq__