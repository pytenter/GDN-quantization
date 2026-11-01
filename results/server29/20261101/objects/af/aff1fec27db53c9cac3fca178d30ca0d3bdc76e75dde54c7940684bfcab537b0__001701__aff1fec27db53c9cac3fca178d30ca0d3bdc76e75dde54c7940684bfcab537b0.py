def __create_fn__(_type_dim, _type_placements, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,dim:_type_dim,placements:_type_placements)->_return_type:
  self.dim=dim
  self.placements=placements
  self.__post_init__()
 return __init__