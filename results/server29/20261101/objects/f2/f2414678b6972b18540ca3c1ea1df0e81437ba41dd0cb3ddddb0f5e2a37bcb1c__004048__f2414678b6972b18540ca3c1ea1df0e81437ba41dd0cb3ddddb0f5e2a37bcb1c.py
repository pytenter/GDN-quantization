def __create_fn__(_type_compute_environment, _type_distributed_type, _type_mixed_precision, _type_use_cpu, _type_debug, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,compute_environment:_type_compute_environment,distributed_type:_type_distributed_type,mixed_precision:_type_mixed_precision,use_cpu:_type_use_cpu,debug:_type_debug)->_return_type:
  self.compute_environment=compute_environment
  self.distributed_type=distributed_type
  self.mixed_precision=mixed_precision
  self.use_cpu=use_cpu
  self.debug=debug
  self.__post_init__()
 return __init__