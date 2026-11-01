def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.cp_comm_strategy,)==(other.cp_comm_strategy,)
  return NotImplemented
 return __eq__