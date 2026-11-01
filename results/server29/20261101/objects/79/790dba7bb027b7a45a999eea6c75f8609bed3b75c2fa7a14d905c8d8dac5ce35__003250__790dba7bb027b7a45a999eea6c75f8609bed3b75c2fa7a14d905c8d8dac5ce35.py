def __create_fn__(_type_params, _type_constant_offset, _type_order, _type_mask_vars, _type_broadcast_shape, _type_broadcasting_dims, _type_final_shape, _type__boundary_check, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt__boundary_check, _return_type):
 def __init__(self,params:_type_params,constant_offset:_type_constant_offset,order:_type_order,mask_vars:_type_mask_vars,broadcast_shape:_type_broadcast_shape,broadcasting_dims:_type_broadcasting_dims,final_shape:_type_final_shape,_boundary_check:_type__boundary_check=_dflt__boundary_check)->_return_type:
  self.params=params
  self.constant_offset=constant_offset
  self.order=order
  self.mask_vars=mask_vars
  self.broadcast_shape=broadcast_shape
  self.broadcasting_dims=broadcasting_dims
  self.final_shape=final_shape
  self._boundary_check=_boundary_check
 return __init__