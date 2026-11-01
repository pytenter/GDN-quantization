def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.compute_environment,self.distributed_type,self.mixed_precision,self.use_cpu,self.debug,)==(other.compute_environment,other.distributed_type,other.mixed_precision,other.use_cpu,other.debug,)
  return NotImplemented
 return __eq__