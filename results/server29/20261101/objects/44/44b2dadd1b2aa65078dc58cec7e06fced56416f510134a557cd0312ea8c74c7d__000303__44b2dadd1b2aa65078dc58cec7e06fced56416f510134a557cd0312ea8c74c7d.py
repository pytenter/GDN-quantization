def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.param_dtype,self.reduce_dtype,self.buffer_dtype,)==(other.param_dtype,other.reduce_dtype,other.buffer_dtype,)
  return NotImplemented
 return __eq__