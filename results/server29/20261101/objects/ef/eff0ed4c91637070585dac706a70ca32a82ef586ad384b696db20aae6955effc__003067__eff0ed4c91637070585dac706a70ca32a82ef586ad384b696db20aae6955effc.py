def __create_fn__(_type_name, _type_mpi_buffer, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_mpi_buffer, _return_type):
 def __init__(self,name:_type_name,mpi_buffer:_type_mpi_buffer=_HAS_DEFAULT_FACTORY)->_return_type:
  self.name=name
  self.mpi_buffer=_dflt_mpi_buffer() if mpi_buffer is _HAS_DEFAULT_FACTORY else mpi_buffer
 return __init__