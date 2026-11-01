def __create_fn__(_type_title, _type_status, _type_num, _type_repo_id, _type_repo_type, _type_author, _type_is_pull_request, _type_created_at, _type_endpoint, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,title:_type_title,status:_type_status,num:_type_num,repo_id:_type_repo_id,repo_type:_type_repo_type,author:_type_author,is_pull_request:_type_is_pull_request,created_at:_type_created_at,endpoint:_type_endpoint)->_return_type:
  self.title=title
  self.status=status
  self.num=num
  self.repo_id=repo_id
  self.repo_type=repo_type
  self.author=author
  self.is_pull_request=is_pull_request
  self.created_at=created_at
  self.endpoint=endpoint
 return __init__