def __create_fn__(_type_dtype, _type_shape, _type_stride, _type_device, _type_layout, _type_memory_format, _type_storage_offset, _type_storage_bytes, _type_requires_grad, _type_is_quantized, _type_is_conj, _type_is_neg, _type_is_inference, _type_is_sparse, _type_is_coalesced, _type_dense_dim, _type_sparse_dim, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,dtype:_type_dtype,shape:_type_shape,stride:_type_stride,device:_type_device,layout:_type_layout,memory_format:_type_memory_format,storage_offset:_type_storage_offset,storage_bytes:_type_storage_bytes,requires_grad:_type_requires_grad,is_quantized:_type_is_quantized,is_conj:_type_is_conj,is_neg:_type_is_neg,is_inference:_type_is_inference,is_sparse:_type_is_sparse,is_coalesced:_type_is_coalesced,dense_dim:_type_dense_dim,sparse_dim:_type_sparse_dim)->_return_type:
  self.dtype=dtype
  self.shape=shape
  self.stride=stride
  self.device=device
  self.layout=layout
  self.memory_format=memory_format
  self.storage_offset=storage_offset
  self.storage_bytes=storage_bytes
  self.requires_grad=requires_grad
  self.is_quantized=is_quantized
  self.is_conj=is_conj
  self.is_neg=is_neg
  self.is_inference=is_inference
  self.is_sparse=is_sparse
  self.is_coalesced=is_coalesced
  self.dense_dim=dense_dim
  self.sparse_dim=sparse_dim
 return __init__