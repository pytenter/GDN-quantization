def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(dp_replicate_size={self.dp_replicate_size!r}, dp_shard_size={self.dp_shard_size!r}, tp_size={self.tp_size!r}, cp_size={self.cp_size!r}, cp_backend={self.cp_backend!r}, sp_size={self.sp_size!r}, sp_backend={self.sp_backend!r}, tp_handler={self.tp_handler!r}, cp_handler={self.cp_handler!r}, sp_handler={self.sp_handler!r})"
 return __repr__