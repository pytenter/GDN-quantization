def __create_fn__(_type_path, _type_size, _type_blob_id, _type_lfs, _type_last_commit, _type_security, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_lfs, _dflt_last_commit, _dflt_security, _return_type):
 def __init__(self,path:_type_path,size:_type_size,blob_id:_type_blob_id,lfs:_type_lfs=_dflt_lfs,last_commit:_type_last_commit=_dflt_last_commit,security:_type_security=_dflt_security)->_return_type:
  self.path=path
  self.size=size
  self.blob_id=blob_id
  self.lfs=lfs
  self.last_commit=last_commit
  self.security=security
 return __init__