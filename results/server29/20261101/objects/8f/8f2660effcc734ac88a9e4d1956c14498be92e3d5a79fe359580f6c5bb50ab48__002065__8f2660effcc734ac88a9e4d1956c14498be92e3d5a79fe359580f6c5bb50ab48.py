def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.repository_uri,self.as_of_time_utc,self.branch,self.mapped_to,self.properties,self.revision_id,self.revision_tag,)==(other.repository_uri,other.as_of_time_utc,other.branch,other.mapped_to,other.properties,other.revision_id,other.revision_tag,)
  return NotImplemented
 return __eq__