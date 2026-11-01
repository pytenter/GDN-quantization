def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dim,self.broadcast_buffers,self.bucket_cap_mb,self.find_unused_parameters,self.check_reduction,self.gradient_as_bucket_view,self.static_graph,self.comm_hook,self.comm_wrapper,self.comm_state_option,)==(other.dim,other.broadcast_buffers,other.bucket_cap_mb,other.find_unused_parameters,other.check_reduction,other.gradient_as_bucket_view,other.static_graph,other.comm_hook,other.comm_wrapper,other.comm_state_option,)
  return NotImplemented
 return __eq__