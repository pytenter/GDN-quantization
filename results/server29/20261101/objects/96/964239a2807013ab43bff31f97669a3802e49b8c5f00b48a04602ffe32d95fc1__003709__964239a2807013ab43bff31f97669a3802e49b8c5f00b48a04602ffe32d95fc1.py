def __create_fn__():
 def __hash__(self):
  return hash((self.commit_hash,self.snapshot_path,self.size_on_disk,self.files,self.refs,self.last_modified,))
 return __hash__