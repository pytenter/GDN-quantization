def __create_fn__(_type_size, _type_timestamp, _type_should_ignore, _type_sha256, _type_upload_mode, _type_remote_oid, _type_is_uploaded, _type_is_committed, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_timestamp, _dflt_should_ignore, _dflt_sha256, _dflt_upload_mode, _dflt_remote_oid, _dflt_is_uploaded, _dflt_is_committed, _return_type):
 def __init__(self,size:_type_size,timestamp:_type_timestamp=_dflt_timestamp,should_ignore:_type_should_ignore=_dflt_should_ignore,sha256:_type_sha256=_dflt_sha256,upload_mode:_type_upload_mode=_dflt_upload_mode,remote_oid:_type_remote_oid=_dflt_remote_oid,is_uploaded:_type_is_uploaded=_dflt_is_uploaded,is_committed:_type_is_committed=_dflt_is_committed)->_return_type:
  self.size=size
  self.timestamp=timestamp
  self.should_ignore=should_ignore
  self.sha256=sha256
  self.upload_mode=upload_mode
  self.remote_oid=remote_oid
  self.is_uploaded=is_uploaded
  self.is_committed=is_committed
 return __init__