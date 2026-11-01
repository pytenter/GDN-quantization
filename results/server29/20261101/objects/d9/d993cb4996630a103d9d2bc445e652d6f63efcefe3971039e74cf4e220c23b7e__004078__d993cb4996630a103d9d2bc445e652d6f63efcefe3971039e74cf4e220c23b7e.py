def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.is_causal,self.sliding_window,)==(other.is_causal,other.sliding_window,)
  return NotImplemented
 return __eq__