def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.file_name,self.file_path,self.blob_path,self.size_on_disk,self.blob_last_accessed,self.blob_last_modified,)==(other.file_name,other.file_path,other.blob_path,other.size_on_disk,other.blob_last_accessed,other.blob_last_modified,)
  return NotImplemented
 return __eq__