def __create_fn__(_type_path_in_repo, _type_is_folder, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_is_folder, _return_type):
 def __init__(self,path_in_repo:_type_path_in_repo,is_folder:_type_is_folder=_dflt_is_folder)->_return_type:
  self.path_in_repo=path_in_repo
  self.is_folder=is_folder
  self.__post_init__()
 return __init__