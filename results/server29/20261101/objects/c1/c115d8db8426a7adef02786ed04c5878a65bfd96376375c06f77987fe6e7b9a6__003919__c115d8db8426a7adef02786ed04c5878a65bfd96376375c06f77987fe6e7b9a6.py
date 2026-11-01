def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.file_oid,self.filename,self.oid,self.pushed_at,self.ref,self.size,)==(other.file_oid,other.filename,other.oid,other.pushed_at,other.ref,other.size,)
  return NotImplemented
 return __eq__