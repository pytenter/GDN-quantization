def __create_fn__(_type_rank, _type_world_size, _type_local_rank, _type_local_world_size, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_rank, _dflt_world_size, _dflt_local_rank, _dflt_local_world_size, _return_type):
 def __init__(self,rank:_type_rank=_dflt_rank,world_size:_type_world_size=_dflt_world_size,local_rank:_type_local_rank=_dflt_local_rank,local_world_size:_type_local_world_size=_dflt_local_world_size)->_return_type:
  self.rank=rank
  self.world_size=world_size
  self.local_rank=local_rank
  self.local_world_size=local_world_size
 return __init__