def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.path,self.tree_id,self.last_commit,)==(other.path,other.tree_id,other.last_commit,)
  return NotImplemented
 return __eq__