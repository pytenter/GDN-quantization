def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.own_mark_name_mapping,)==(other.own_mark_name_mapping,)
  return NotImplemented
 return __eq__