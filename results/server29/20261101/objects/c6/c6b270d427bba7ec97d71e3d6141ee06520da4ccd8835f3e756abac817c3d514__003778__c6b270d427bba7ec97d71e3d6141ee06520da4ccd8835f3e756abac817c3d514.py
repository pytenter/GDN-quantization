def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.src_path_in_repo,self.path_in_repo,self.src_revision,self._src_oid,self._dest_oid,)==(other.src_path_in_repo,other.path_in_repo,other.src_revision,other._src_oid,other._dest_oid,)
  return NotImplemented
 return __eq__