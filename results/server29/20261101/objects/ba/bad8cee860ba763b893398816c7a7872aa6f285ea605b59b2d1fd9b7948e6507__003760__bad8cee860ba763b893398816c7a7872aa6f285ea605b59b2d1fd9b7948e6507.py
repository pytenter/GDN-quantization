def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.filename,self.commit_hash,self.etag,self.timestamp,)==(other.filename,other.commit_hash,other.etag,other.timestamp,)
  return NotImplemented
 return __eq__