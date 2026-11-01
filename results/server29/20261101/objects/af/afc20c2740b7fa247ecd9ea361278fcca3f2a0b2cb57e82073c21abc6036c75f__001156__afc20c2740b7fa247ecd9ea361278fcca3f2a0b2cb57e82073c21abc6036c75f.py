def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.f,self.args,self.kwargs,self.tracked_fakes,self.name,)==(other.f,other.args,other.kwargs,other.tracked_fakes,other.name,)
  return NotImplemented
 return __eq__