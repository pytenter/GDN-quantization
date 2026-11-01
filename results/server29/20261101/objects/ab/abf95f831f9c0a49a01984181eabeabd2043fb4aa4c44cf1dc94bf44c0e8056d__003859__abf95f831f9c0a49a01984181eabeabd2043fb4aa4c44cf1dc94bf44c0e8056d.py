def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.commit_url,self.commit_message,self.commit_description,self.oid,self.pr_url,self.repo_url,self.pr_revision,self.pr_num,self._url,)==(other.commit_url,other.commit_message,other.commit_description,other.oid,other.pr_url,other.repo_url,other.pr_revision,other.pr_num,other._url,)
  return NotImplemented
 return __eq__