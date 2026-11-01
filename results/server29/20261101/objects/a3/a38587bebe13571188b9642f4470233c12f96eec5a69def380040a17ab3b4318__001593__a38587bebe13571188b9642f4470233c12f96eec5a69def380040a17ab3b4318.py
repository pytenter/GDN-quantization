def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.present_in_generic_dict,self.l1_key,self.l2_key,)==(other.present_in_generic_dict,other.l1_key,other.l2_key,)
  return NotImplemented
 return __eq__