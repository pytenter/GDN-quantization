def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self._result,self.excinfo,self.start,self.stop,self.duration,self.when,)==(other._result,other.excinfo,other.start,other.stop,other.duration,other.when,)
  return NotImplemented
 return __eq__