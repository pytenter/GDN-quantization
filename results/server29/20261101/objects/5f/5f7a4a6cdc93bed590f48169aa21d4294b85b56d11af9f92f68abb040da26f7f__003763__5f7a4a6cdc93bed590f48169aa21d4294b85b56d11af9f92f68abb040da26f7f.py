def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.size,self.timestamp,self.should_ignore,self.sha256,self.upload_mode,self.remote_oid,self.is_uploaded,self.is_committed,)==(other.size,other.timestamp,other.should_ignore,other.sha256,other.upload_mode,other.remote_oid,other.is_uploaded,other.is_committed,)
  return NotImplemented
 return __eq__