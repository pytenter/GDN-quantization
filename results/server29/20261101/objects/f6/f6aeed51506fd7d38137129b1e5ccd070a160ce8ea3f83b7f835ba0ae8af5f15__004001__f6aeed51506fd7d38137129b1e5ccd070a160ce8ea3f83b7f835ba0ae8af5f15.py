def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.project_dir,self.logging_dir,self.automatic_checkpoint_naming,self.total_limit,self.iteration,self.save_on_each_node,)==(other.project_dir,other.logging_dir,other.automatic_checkpoint_naming,other.total_limit,other.iteration,other.save_on_each_node,)
  return NotImplemented
 return __eq__