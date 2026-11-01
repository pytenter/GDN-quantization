def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.use_autocast_during_eval,self.margin,self.interval,self.fp8_format,self.amax_history_len,self.amax_compute_algo,self.override_linear_precision,self.use_mxfp8_block_scaling,)==(other.use_autocast_during_eval,other.margin,other.interval,other.fp8_format,other.amax_history_len,other.amax_compute_algo,other.override_linear_precision,other.use_mxfp8_block_scaling,)
  return NotImplemented
 return __eq__