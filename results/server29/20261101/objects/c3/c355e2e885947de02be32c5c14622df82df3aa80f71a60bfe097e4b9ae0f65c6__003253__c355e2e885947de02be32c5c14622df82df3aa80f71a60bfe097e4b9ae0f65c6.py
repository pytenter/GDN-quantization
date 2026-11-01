def __create_fn__(_type_shape, _type_block_shape, _type_strides, _type_offsets, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_shape, _dflt_block_shape, _dflt_strides, _dflt_offsets, _return_type):
 def __init__(self,shape:_type_shape=_HAS_DEFAULT_FACTORY,block_shape:_type_block_shape=_HAS_DEFAULT_FACTORY,strides:_type_strides=_HAS_DEFAULT_FACTORY,offsets:_type_offsets=_HAS_DEFAULT_FACTORY)->_return_type:
  self.shape=_dflt_shape() if shape is _HAS_DEFAULT_FACTORY else shape
  self.block_shape=_dflt_block_shape() if block_shape is _HAS_DEFAULT_FACTORY else block_shape
  self.strides=_dflt_strides() if strides is _HAS_DEFAULT_FACTORY else strides
  self.offsets=_dflt_offsets() if offsets is _HAS_DEFAULT_FACTORY else offsets
 return __init__