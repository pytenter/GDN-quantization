def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(rank={self.rank!r}, world_size={self.world_size!r}, local_rank={self.local_rank!r}, local_world_size={self.local_world_size!r})"
 return __repr__