def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.commit_id,self.authors,self.created_at,self.title,self.message,self.formatted_title,self.formatted_message,)==(other.commit_id,other.authors,other.created_at,other.title,other.message,other.formatted_title,other.formatted_message,)
  return NotImplemented
 return __eq__