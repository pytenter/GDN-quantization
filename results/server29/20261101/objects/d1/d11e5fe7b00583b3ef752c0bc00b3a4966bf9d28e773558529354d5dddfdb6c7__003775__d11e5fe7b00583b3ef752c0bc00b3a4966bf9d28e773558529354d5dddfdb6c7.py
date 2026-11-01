def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.path_in_repo,self.is_folder,)==(other.path_in_repo,other.is_folder,)
  return NotImplemented
 return __eq__