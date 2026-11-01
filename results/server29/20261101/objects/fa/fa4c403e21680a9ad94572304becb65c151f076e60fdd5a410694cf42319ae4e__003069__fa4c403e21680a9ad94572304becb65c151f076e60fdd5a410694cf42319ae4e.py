def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.mpi_buffer,)==(other.name,other.mpi_buffer,)
  return NotImplemented
 return __eq__