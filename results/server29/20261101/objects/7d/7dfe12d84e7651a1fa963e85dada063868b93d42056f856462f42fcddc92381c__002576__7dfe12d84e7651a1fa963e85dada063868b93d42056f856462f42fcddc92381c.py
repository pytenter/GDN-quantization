def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.fusible_ops,self.compute_intensive_ops,self.random_ops,self.view_ops,self.recomputable_ops,)==(other.fusible_ops,other.compute_intensive_ops,other.random_ops,other.view_ops,other.recomputable_ops,)
  return NotImplemented
 return __eq__