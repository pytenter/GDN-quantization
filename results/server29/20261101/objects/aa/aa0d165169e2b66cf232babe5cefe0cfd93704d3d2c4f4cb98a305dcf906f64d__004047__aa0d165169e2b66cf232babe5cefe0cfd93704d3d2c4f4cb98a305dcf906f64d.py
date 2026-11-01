def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dp_replicate_size,self.dp_shard_size,self.tp_size,self.cp_size,self.cp_backend,self.sp_size,self.sp_backend,self.tp_handler,self.cp_handler,self.sp_handler,)==(other.dp_replicate_size,other.dp_shard_size,other.tp_size,other.cp_size,other.cp_backend,other.sp_size,other.sp_backend,other.tp_handler,other.cp_handler,other.sp_handler,)
  return NotImplemented
 return __eq__