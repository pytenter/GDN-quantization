def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.type_promotion_kind,self.override_return_dtype,)==(other.type_promotion_kind,other.override_return_dtype,)
  return NotImplemented
 return __eq__