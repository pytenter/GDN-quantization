def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.body,self.template_mask,self.template_out,self.compute,self.indexing_code,self.loads,self.stores,self.ops_handler,self.range_trees,)==(other.body,other.template_mask,other.template_out,other.compute,other.indexing_code,other.loads,other.stores,other.ops_handler,other.range_trees,)
  return NotImplemented
 return __eq__