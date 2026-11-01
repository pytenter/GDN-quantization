def __create_fn__(_type_commit_url, _type_commit_message, _type_commit_description, _type_oid, _type_pr_url, _type_repo_url, _type_pr_revision, _type_pr_num, _type__url, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_pr_url, _dflt__url, _return_type):
 def __init__(self,commit_url:_type_commit_url,commit_message:_type_commit_message,commit_description:_type_commit_description,oid:_type_oid,pr_url:_type_pr_url=_dflt_pr_url,_url:_type__url=_dflt__url)->_return_type:
  self.commit_url=commit_url
  self.commit_message=commit_message
  self.commit_description=commit_description
  self.oid=oid
  self.pr_url=pr_url
  self._url=_url
  self.__post_init__()
 return __init__