def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.rfilename,self.size,self.blob_id,self.lfs,)==(other.rfilename,other.size,other.blob_id,other.lfs,)
  return NotImplemented
 return __eq__