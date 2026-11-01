def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.arg,self.custom_obj_name,)==(other.arg,other.custom_obj_name,)
  return NotImplemented
 return __eq__