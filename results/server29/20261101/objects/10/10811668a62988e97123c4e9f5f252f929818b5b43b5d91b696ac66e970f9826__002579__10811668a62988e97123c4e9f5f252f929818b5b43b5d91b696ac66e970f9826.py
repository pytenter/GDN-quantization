def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.inputs,self._required_fw_nodes,self.required_bw_nodes,self.unclaimed_nodes,self.fw_order,)==(other.inputs,other._required_fw_nodes,other.required_bw_nodes,other.unclaimed_nodes,other.fw_order,)
  return NotImplemented
 return __eq__