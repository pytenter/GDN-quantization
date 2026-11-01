def __create_fn__(_type_dp_replicate_size, _type_dp_shard_size, _type_tp_size, _type_cp_size, _type_cp_backend, _type_sp_size, _type_sp_backend, _type_tp_handler, _type_cp_handler, _type_sp_handler, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_dp_replicate_size, _dflt_dp_shard_size, _dflt_tp_size, _dflt_cp_size, _dflt_cp_backend, _dflt_sp_size, _dflt_sp_backend, _dflt_tp_handler, _dflt_cp_handler, _dflt_sp_handler, _return_type):
 def __init__(self,dp_replicate_size:_type_dp_replicate_size=_dflt_dp_replicate_size,dp_shard_size:_type_dp_shard_size=_dflt_dp_shard_size,tp_size:_type_tp_size=_dflt_tp_size,cp_size:_type_cp_size=_dflt_cp_size,cp_backend:_type_cp_backend=_dflt_cp_backend,sp_size:_type_sp_size=_dflt_sp_size,sp_backend:_type_sp_backend=_dflt_sp_backend,tp_handler:_type_tp_handler=_dflt_tp_handler,cp_handler:_type_cp_handler=_dflt_cp_handler,sp_handler:_type_sp_handler=_dflt_sp_handler)->_return_type:
  self.dp_replicate_size=dp_replicate_size
  self.dp_shard_size=dp_shard_size
  self.tp_size=tp_size
  self.cp_size=cp_size
  self.cp_backend=cp_backend
  self.sp_size=sp_size
  self.sp_backend=sp_backend
  self.tp_handler=tp_handler
  self.cp_handler=cp_handler
  self.sp_handler=sp_handler
  self.__post_init__()
 return __init__