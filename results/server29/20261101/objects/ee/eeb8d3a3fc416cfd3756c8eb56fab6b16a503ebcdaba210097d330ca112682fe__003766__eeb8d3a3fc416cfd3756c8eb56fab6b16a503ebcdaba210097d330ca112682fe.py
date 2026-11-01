def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.commit_hash,self.etag,self.location,self.size,self.xet_file_data,)==(other.commit_hash,other.etag,other.location,other.size,other.xet_file_data,)
  return NotImplemented
 return __eq__