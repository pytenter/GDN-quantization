def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.dtype,self.shape,self.stride,self.device,self.layout,self.memory_format,self.storage_offset,self.storage_bytes,self.requires_grad,self.is_quantized,self.is_conj,self.is_neg,self.is_inference,self.is_sparse,self.is_coalesced,self.dense_dim,self.sparse_dim,)==(other.dtype,other.shape,other.stride,other.device,other.layout,other.memory_format,other.storage_offset,other.storage_bytes,other.requires_grad,other.is_quantized,other.is_conj,other.is_neg,other.is_inference,other.is_sparse,other.is_coalesced,other.dense_dim,other.sparse_dim,)
  return NotImplemented
 return __eq__