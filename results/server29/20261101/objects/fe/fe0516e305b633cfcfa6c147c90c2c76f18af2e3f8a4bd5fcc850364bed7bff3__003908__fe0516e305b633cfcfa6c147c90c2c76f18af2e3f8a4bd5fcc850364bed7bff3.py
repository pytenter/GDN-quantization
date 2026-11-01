def __create_fn__(_type_avatar_url, _type_name, _type_fullname, _type_details, _type_is_verified, _type_is_following, _type_num_users, _type_num_models, _type_num_spaces, _type_num_datasets, _type_num_followers, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_details, _dflt_is_verified, _dflt_is_following, _dflt_num_users, _dflt_num_models, _dflt_num_spaces, _dflt_num_datasets, _dflt_num_followers, _return_type):
 def __init__(self,avatar_url:_type_avatar_url,name:_type_name,fullname:_type_fullname,details:_type_details=_dflt_details,is_verified:_type_is_verified=_dflt_is_verified,is_following:_type_is_following=_dflt_is_following,num_users:_type_num_users=_dflt_num_users,num_models:_type_num_models=_dflt_num_models,num_spaces:_type_num_spaces=_dflt_num_spaces,num_datasets:_type_num_datasets=_dflt_num_datasets,num_followers:_type_num_followers=_dflt_num_followers)->_return_type:
  self.avatar_url=avatar_url
  self.name=name
  self.fullname=fullname
  self.details=details
  self.is_verified=is_verified
  self.is_following=is_following
  self.num_users=num_users
  self.num_models=num_models
  self.num_spaces=num_spaces
  self.num_datasets=num_datasets
  self.num_followers=num_followers
 return __init__