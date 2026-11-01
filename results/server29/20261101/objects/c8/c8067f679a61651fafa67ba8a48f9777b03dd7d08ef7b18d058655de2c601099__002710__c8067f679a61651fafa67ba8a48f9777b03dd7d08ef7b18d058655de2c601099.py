def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arg,self.user_input_name,)==(other.arg,other.user_input_name,)
  return NotImplemented
 return __eq__