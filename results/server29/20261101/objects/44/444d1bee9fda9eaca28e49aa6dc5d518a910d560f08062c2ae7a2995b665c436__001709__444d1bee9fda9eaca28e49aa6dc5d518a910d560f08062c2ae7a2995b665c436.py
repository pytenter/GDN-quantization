def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(local_rank={self.local_rank!r}, global_rank={self.global_rank!r}, local_world_size={self.local_world_size!r}, global_world_size={self.global_world_size!r})"
 return __repr__