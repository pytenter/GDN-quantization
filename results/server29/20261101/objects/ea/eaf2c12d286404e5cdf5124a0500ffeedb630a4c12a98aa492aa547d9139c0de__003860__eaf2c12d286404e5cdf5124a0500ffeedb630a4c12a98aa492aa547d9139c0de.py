def __create_fn__(_type_username, _type_fullname, _type_email, _type_timestamp, _type_status, _type_fields, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_fields, _return_type):
 def __init__(self,username:_type_username,fullname:_type_fullname,email:_type_email,timestamp:_type_timestamp,status:_type_status,fields:_type_fields=_dflt_fields)->_return_type:
  self.username=username
  self.fullname=fullname
  self.email=email
  self.timestamp=timestamp
  self.status=status
  self.fields=fields
 return __init__