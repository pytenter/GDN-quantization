def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.args,self.outs,)==(other.args,other.outs,)
  return NotImplemented
 return __eq__