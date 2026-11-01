def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.group,self.cu_seqlens,self.cu_seqlens_cpu,self.is_last_rank,self.pre_num_ranks,self.is_first_rank,self.post_num_ranks,self.conv1d_kernel_size,self.pre_num_conv_tokens,)==(other.group,other.cu_seqlens,other.cu_seqlens_cpu,other.is_last_rank,other.pre_num_ranks,other.is_first_rank,other.post_num_ranks,other.conv1d_kernel_size,other.pre_num_conv_tokens,)
  return NotImplemented
 return __eq__