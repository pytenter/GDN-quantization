def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.contents,self.description,self.encoding,self.hashes,self.last_modified_time_utc,self.length,self.location,self.mime_type,self.offset,self.parent_index,self.properties,self.roles,self.source_language,)==(other.contents,other.description,other.encoding,other.hashes,other.last_modified_time_utc,other.length,other.location,other.mime_type,other.offset,other.parent_index,other.properties,other.roles,other.source_language,)
  return NotImplemented
 return __eq__