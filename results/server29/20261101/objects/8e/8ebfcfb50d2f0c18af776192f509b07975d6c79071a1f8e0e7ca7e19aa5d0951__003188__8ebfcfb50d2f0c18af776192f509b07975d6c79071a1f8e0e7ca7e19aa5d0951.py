def __create_fn__(_type_inductor_meta, _type_mode, _type_prefix, _type_x_grid, _type_y_grid, _type_z_grid, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_mode, _dflt_prefix, _dflt_x_grid, _dflt_y_grid, _dflt_z_grid, _return_type):
 def __init__(self,inductor_meta:_type_inductor_meta,mode:_type_mode=_dflt_mode,prefix:_type_prefix=_dflt_prefix,x_grid:_type_x_grid=_dflt_x_grid,y_grid:_type_y_grid=_dflt_y_grid,z_grid:_type_z_grid=_dflt_z_grid)->_return_type:
  self.inductor_meta=inductor_meta
  self.mode=mode
  self.prefix=prefix
  self.x_grid=x_grid
  self.y_grid=y_grid
  self.z_grid=z_grid
  self.__post_init__()
 return __init__