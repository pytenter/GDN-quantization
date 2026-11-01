def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.path,self.size,self.blob_id,self.lfs,self.last_commit,self.security,)==(other.path,other.size,other.blob_id,other.lfs,other.last_commit,other.security,)
  return NotImplemented
 return __eq__