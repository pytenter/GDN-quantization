def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.branches,self.converts,self.tags,self.pull_requests,)==(other.branches,other.converts,other.tags,other.pull_requests,)
  return NotImplemented
 return __eq__