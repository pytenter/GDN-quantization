def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.is_acc,self.nodes,self.device_ordinal,)==(other.is_acc,other.nodes,other.device_ordinal,)
  return NotImplemented
 return __eq__