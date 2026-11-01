def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.repo_id,self.repo_type,self.repo_path,self.size_on_disk,self.nb_files,self.revisions,self.last_accessed,self.last_modified,)==(other.repo_id,other.repo_type,other.repo_path,other.size_on_disk,other.nb_files,other.revisions,other.last_accessed,other.last_modified,)
  return NotImplemented
 return __eq__