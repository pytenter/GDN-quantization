def __create_fn__():
 def __hash__(self):
  return hash((self.repo_id,self.repo_type,self.repo_path,self.size_on_disk,self.nb_files,self.revisions,self.last_accessed,self.last_modified,))
 return __hash__