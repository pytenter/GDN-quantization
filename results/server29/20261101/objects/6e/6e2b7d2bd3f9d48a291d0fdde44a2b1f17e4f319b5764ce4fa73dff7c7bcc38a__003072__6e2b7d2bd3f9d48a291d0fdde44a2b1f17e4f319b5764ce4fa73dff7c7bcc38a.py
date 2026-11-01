def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.scheduler,self.node,self.defining_op,self.users,self.mpi_buffer,)==(other.scheduler,other.node,other.defining_op,other.users,other.mpi_buffer,)
  return NotImplemented
 return __eq__