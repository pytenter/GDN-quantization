def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.frame_exec_strategy,self.apply_to_code,self.guarded_code,)==(other.frame_exec_strategy,other.apply_to_code,other.guarded_code,)
  return NotImplemented
 return __eq__