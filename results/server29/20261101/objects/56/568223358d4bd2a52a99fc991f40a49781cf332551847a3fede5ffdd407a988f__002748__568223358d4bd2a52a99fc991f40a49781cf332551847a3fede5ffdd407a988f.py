def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.tensorPaths,self.program,self.delegates,self.deviceAllocationMap,self.constantPaths,)==(other.name,other.tensorPaths,other.program,other.delegates,other.deviceAllocationMap,other.constantPaths,)
  return NotImplemented
 return __eq__