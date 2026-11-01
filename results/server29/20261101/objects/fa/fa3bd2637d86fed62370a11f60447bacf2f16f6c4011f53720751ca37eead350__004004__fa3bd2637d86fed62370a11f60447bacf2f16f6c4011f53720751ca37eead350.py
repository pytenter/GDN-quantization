def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.num_steps,self.adjust_scheduler,self.sync_with_dataloader,self.sync_each_batch,)==(other.num_steps,other.adjust_scheduler,other.sync_with_dataloader,other.sync_each_batch,)
  return NotImplemented
 return __eq__