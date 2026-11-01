def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(title={self.title!r}, status={self.status!r}, num={self.num!r}, repo_id={self.repo_id!r}, repo_type={self.repo_type!r}, author={self.author!r}, is_pull_request={self.is_pull_request!r}, created_at={self.created_at!r}, endpoint={self.endpoint!r}, events={self.events!r}, conflicting_files={self.conflicting_files!r}, target_branch={self.target_branch!r}, merge_commit_oid={self.merge_commit_oid!r}, diff={self.diff!r})"
 return __repr__