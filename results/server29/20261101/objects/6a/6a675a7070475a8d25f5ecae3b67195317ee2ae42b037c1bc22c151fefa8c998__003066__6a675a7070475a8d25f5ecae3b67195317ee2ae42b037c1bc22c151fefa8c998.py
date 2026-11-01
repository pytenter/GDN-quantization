def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.index,self.size,self.pred_buffers,self.pred_nodes,self.succ_nodes,)==(other.index,other.size,other.pred_buffers,other.pred_nodes,other.succ_nodes,)
  return NotImplemented
 return __eq__