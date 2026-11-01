def __create_fn__(_type_op, _type_args_schema, _type_kwargs_schema, _type_schema_info, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_schema_info, _return_type):
 def __init__(self,op:_type_op,args_schema:_type_args_schema,kwargs_schema:_type_kwargs_schema,schema_info:_type_schema_info=_dflt_schema_info)->_return_type:
  self.op=op
  self.args_schema=args_schema
  self.kwargs_schema=kwargs_schema
  self.schema_info=schema_info
  self.__post_init__()
 return __init__