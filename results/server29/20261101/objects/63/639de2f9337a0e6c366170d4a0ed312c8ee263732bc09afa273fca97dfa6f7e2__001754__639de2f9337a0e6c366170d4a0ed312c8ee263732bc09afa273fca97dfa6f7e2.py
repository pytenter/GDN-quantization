def __create_fn__(_type_compute_mesh, _type_schema, _type_flat_args_schema, _type_local_args, _type_local_kwargs, _type_args_tree_spec, _type_output_sharding, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_args_tree_spec, _dflt_output_sharding, _return_type):
 def __init__(self,compute_mesh:_type_compute_mesh,schema:_type_schema,flat_args_schema:_type_flat_args_schema,local_args:_type_local_args,local_kwargs:_type_local_kwargs,args_tree_spec:_type_args_tree_spec=_dflt_args_tree_spec,output_sharding:_type_output_sharding=_dflt_output_sharding)->_return_type:
  self.compute_mesh=compute_mesh
  self.schema=schema
  self.flat_args_schema=flat_args_schema
  self.local_args=local_args
  self.local_kwargs=local_kwargs
  self.args_tree_spec=args_tree_spec
  self.output_sharding=output_sharding
 return __init__