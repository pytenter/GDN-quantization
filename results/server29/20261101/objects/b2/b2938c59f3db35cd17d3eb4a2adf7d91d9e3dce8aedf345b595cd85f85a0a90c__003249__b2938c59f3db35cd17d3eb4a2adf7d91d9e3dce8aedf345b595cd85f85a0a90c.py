def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.index_str,self.mask_vars,self.expand_str,self._has_rindex,self.index,)==(other.index_str,other.mask_vars,other.expand_str,other._has_rindex,other.index,)
  return NotImplemented
 return __eq__