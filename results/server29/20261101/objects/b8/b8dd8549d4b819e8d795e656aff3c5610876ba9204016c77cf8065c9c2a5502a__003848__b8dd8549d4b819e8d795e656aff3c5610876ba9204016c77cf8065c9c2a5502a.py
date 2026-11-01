def __create_fn__(_type_safe, _type_status, _type_av_scan, _type_pickle_import_scan, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,safe:_type_safe,status:_type_status,av_scan:_type_av_scan,pickle_import_scan:_type_pickle_import_scan)->_return_type:
  self.safe=safe
  self.status=status
  self.av_scan=av_scan
  self.pickle_import_scan=pickle_import_scan
  self.__post_init__()
 return __init__