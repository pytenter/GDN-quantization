def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.input_dim,self.group_shape,self.split_id,)==(other.input_dim,other.group_shape,other.split_id,)
  return NotImplemented
 return __eq__