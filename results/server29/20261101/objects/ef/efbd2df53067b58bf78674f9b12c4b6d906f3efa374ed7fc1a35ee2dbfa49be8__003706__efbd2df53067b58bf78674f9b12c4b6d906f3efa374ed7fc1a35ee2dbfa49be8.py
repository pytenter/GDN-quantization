def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.commit_hash,self.snapshot_path,self.size_on_disk,self.files,self.refs,self.last_modified,)==(other.commit_hash,other.snapshot_path,other.size_on_disk,other.files,other.refs,other.last_modified,)
  return NotImplemented
 return __eq__