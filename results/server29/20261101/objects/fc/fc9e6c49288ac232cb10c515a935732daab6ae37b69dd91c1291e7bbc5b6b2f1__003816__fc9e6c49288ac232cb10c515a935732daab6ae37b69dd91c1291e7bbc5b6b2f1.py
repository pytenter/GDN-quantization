def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.title,self.status,self.num,self.repo_id,self.repo_type,self.author,self.is_pull_request,self.created_at,self.endpoint,)==(other.title,other.status,other.num,other.repo_id,other.repo_type,other.author,other.is_pull_request,other.created_at,other.endpoint,)
  return NotImplemented
 return __eq__