def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.left_end_lineno,self.left_end_offset,self.right_start_lineno,self.right_start_offset,)==(other.left_end_lineno,other.left_end_offset,other.right_start_lineno,other.right_start_offset,)
  return NotImplemented
 return __eq__