def __create_fn__(_type_mesh, _type_placements, _type_tensor_meta, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_tensor_meta, _return_type):
 def __init__(self,mesh:_type_mesh,placements:_type_placements,tensor_meta:_type_tensor_meta=_dflt_tensor_meta)->_return_type:
  self.mesh=mesh
  self.placements=placements
  self.tensor_meta=tensor_meta
  self.__post_init__()
 return __init__