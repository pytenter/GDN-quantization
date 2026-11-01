def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.namespace,self.op_name,self.overload,)==(other.namespace,other.op_name,other.overload,)
  return NotImplemented
 return __eq__