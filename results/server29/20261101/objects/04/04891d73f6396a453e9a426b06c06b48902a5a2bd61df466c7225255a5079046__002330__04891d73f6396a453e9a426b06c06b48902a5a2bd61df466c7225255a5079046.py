def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.accepted_params,self.accepts_kwargs,self.default_params,)==(other.accepted_params,other.accepts_kwargs,other.default_params,)
  return NotImplemented
 return __eq__