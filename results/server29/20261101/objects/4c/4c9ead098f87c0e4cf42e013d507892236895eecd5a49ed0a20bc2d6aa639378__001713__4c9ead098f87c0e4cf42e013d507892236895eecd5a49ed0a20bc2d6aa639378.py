def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.plan,self.output_plan,self.return_local_tensor,)==(other.plan,other.output_plan,other.return_local_tensor,)
  return NotImplemented
 return __eq__