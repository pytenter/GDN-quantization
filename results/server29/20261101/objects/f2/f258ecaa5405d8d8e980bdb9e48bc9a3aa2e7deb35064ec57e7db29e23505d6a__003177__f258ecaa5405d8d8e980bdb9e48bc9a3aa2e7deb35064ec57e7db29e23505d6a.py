def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.count_per_thread,self.bytes_per_thread,)==(other.count_per_thread,other.bytes_per_thread,)
  return NotImplemented
 return __eq__