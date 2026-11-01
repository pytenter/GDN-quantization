def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.expected_freed_size,self.blobs,self.refs,self.repos,self.snapshots,)==(other.expected_freed_size,other.blobs,other.refs,other.repos,other.snapshots,)
  return NotImplemented
 return __eq__