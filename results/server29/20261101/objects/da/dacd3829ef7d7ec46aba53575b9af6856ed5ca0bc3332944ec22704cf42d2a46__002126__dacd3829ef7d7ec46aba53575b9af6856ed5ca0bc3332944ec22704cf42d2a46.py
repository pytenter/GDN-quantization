def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.source,self._example,self.pass_arg_as_tensor,self.fake_tensor,self.is_tensor,self.example_strong_ref,)==(other.source,other._example,other.pass_arg_as_tensor,other.fake_tensor,other.is_tensor,other.example_strong_ref,)
  return NotImplemented
 return __eq__