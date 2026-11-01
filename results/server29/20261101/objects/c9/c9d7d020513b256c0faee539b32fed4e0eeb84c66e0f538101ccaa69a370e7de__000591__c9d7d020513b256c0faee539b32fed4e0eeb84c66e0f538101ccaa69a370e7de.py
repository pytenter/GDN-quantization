def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.accumulate_error,self.traverse_method,self.find_all,self.return_intermediate,self.all_outputs,)==(other.accumulate_error,other.traverse_method,other.find_all,other.return_intermediate,other.all_outputs,)
  return NotImplemented
 return __eq__