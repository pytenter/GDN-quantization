def __create_fn__(_type_scheduler, _type_node, _type_defining_op, _type_users, _type_mpi_buffer, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_users, _dflt_mpi_buffer, _return_type):
 def __init__(self,scheduler:_type_scheduler,node:_type_node,defining_op:_type_defining_op,users:_type_users=_HAS_DEFAULT_FACTORY,mpi_buffer:_type_mpi_buffer=_HAS_DEFAULT_FACTORY)->_return_type:
  self.scheduler=scheduler
  self.node=node
  self.defining_op=defining_op
  self.users=_dflt_users() if users is _HAS_DEFAULT_FACTORY else users
  self.mpi_buffer=_dflt_mpi_buffer() if mpi_buffer is _HAS_DEFAULT_FACTORY else mpi_buffer
 return __init__