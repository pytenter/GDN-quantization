def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.path_in_repo,self.path_or_fileobj,self.upload_info,self._upload_mode,self._should_ignore,self._remote_oid,self._is_uploaded,self._is_committed,)==(other.path_in_repo,other.path_or_fileobj,other.upload_info,other._upload_mode,other._should_ignore,other._remote_oid,other._is_uploaded,other._is_committed,)
  return NotImplemented
 return __eq__