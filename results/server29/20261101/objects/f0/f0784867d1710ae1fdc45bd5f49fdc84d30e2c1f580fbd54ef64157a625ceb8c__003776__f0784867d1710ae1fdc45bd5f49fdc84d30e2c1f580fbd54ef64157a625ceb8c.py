def __create_fn__(_type_src_path_in_repo, _type_path_in_repo, _type_src_revision, _type__src_oid, _type__dest_oid, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_src_revision, _dflt__src_oid, _dflt__dest_oid, _return_type):
 def __init__(self,src_path_in_repo:_type_src_path_in_repo,path_in_repo:_type_path_in_repo,src_revision:_type_src_revision=_dflt_src_revision,_src_oid:_type__src_oid=_dflt__src_oid,_dest_oid:_type__dest_oid=_dflt__dest_oid)->_return_type:
  self.src_path_in_repo=src_path_in_repo
  self.path_in_repo=path_in_repo
  self.src_revision=src_revision
  self._src_oid=_src_oid
  self._dest_oid=_dest_oid
  self.__post_init__()
 return __init__