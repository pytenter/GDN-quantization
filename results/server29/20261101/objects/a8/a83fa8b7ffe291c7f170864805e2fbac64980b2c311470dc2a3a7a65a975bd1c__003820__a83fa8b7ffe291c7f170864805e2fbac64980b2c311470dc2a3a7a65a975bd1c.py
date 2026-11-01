def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.title,self.status,self.num,self.repo_id,self.repo_type,self.author,self.is_pull_request,self.created_at,self.endpoint,self.events,self.conflicting_files,self.target_branch,self.merge_commit_oid,self.diff,)==(other.title,other.status,other.num,other.repo_id,other.repo_type,other.author,other.is_pull_request,other.created_at,other.endpoint,other.events,other.conflicting_files,other.target_branch,other.merge_commit_oid,other.diff,)
  return NotImplemented
 return __eq__