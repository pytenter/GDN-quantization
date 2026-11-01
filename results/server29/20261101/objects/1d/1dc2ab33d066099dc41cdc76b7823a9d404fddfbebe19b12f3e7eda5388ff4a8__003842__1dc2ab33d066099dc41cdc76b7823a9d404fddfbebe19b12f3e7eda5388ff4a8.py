def __create_fn__(_type_oid, _type_title, _type_date, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,oid:_type_oid,title:_type_title,date:_type_date)->_return_type:
  self.oid=oid
  self.title=title
  self.date=date
  self.__post_init__()
 return __init__