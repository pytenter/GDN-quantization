def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.stance,self.skip_guard_eval_unsafe,self.backend,)==(other.stance,other.skip_guard_eval_unsafe,other.backend,)
  return NotImplemented
 return __eq__