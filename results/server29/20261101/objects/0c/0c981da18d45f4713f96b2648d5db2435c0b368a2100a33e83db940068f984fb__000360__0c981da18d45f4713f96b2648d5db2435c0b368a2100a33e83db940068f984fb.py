def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.start,self.end,self.queue_depth,)==(other.start,other.end,other.queue_depth,)
  return NotImplemented
 return __eq__