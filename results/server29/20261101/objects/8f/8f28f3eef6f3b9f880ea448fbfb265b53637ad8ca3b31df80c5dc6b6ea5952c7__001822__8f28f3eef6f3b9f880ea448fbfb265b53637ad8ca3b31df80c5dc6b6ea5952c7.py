def __create_fn__(_type_mesh, _type_shard_mesh_dim, _type_replicate_mesh_dim, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_shard_mesh_dim, _dflt_replicate_mesh_dim, _return_type):
 def __init__(self,mesh:_type_mesh,shard_mesh_dim:_type_shard_mesh_dim=_dflt_shard_mesh_dim,replicate_mesh_dim:_type_replicate_mesh_dim=_dflt_replicate_mesh_dim)->_return_type:
  self.mesh=mesh
  self.shard_mesh_dim=shard_mesh_dim
  self.replicate_mesh_dim=replicate_mesh_dim
  self.__post_init__()
 return __init__