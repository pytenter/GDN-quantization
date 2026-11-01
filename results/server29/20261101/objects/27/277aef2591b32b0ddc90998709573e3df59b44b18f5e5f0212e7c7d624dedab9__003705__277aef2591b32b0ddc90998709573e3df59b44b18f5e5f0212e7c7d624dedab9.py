def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(commit_hash={self.commit_hash!r}, snapshot_path={self.snapshot_path!r}, size_on_disk={self.size_on_disk!r}, files={self.files!r}, refs={self.refs!r}, last_modified={self.last_modified!r})"
 return __repr__