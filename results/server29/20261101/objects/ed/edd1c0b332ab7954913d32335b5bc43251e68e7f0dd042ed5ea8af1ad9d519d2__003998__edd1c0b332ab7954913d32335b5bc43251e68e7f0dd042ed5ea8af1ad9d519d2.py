def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.split_batches,self.dispatch_batches,self.even_batches,self.use_seedable_sampler,self.data_seed,self.non_blocking,self.use_stateful_dataloader,)==(other.split_batches,other.dispatch_batches,other.even_batches,other.use_seedable_sampler,other.data_seed,other.non_blocking,other.use_stateful_dataloader,)
  return NotImplemented
 return __eq__