def __create_fn__(_type_username, _type_fullname, _type_avatar_url, _type_details, _type_is_following, _type_is_pro, _type_num_models, _type_num_datasets, _type_num_spaces, _type_num_discussions, _type_num_papers, _type_num_upvotes, _type_num_likes, _type_num_following, _type_num_followers, _type_orgs, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_details, _dflt_is_following, _dflt_is_pro, _dflt_num_models, _dflt_num_datasets, _dflt_num_spaces, _dflt_num_discussions, _dflt_num_papers, _dflt_num_upvotes, _dflt_num_likes, _dflt_num_following, _dflt_num_followers, _dflt_orgs, _return_type):
 def __init__(self,username:_type_username,fullname:_type_fullname,avatar_url:_type_avatar_url,details:_type_details=_dflt_details,is_following:_type_is_following=_dflt_is_following,is_pro:_type_is_pro=_dflt_is_pro,num_models:_type_num_models=_dflt_num_models,num_datasets:_type_num_datasets=_dflt_num_datasets,num_spaces:_type_num_spaces=_dflt_num_spaces,num_discussions:_type_num_discussions=_dflt_num_discussions,num_papers:_type_num_papers=_dflt_num_papers,num_upvotes:_type_num_upvotes=_dflt_num_upvotes,num_likes:_type_num_likes=_dflt_num_likes,num_following:_type_num_following=_dflt_num_following,num_followers:_type_num_followers=_dflt_num_followers,orgs:_type_orgs=_HAS_DEFAULT_FACTORY)->_return_type:
  self.username=username
  self.fullname=fullname
  self.avatar_url=avatar_url
  self.details=details
  self.is_following=is_following
  self.is_pro=is_pro
  self.num_models=num_models
  self.num_datasets=num_datasets
  self.num_spaces=num_spaces
  self.num_discussions=num_discussions
  self.num_papers=num_papers
  self.num_upvotes=num_upvotes
  self.num_likes=num_likes
  self.num_following=num_following
  self.num_followers=num_followers
  self.orgs=_dflt_orgs() if orgs is _HAS_DEFAULT_FACTORY else orgs
 return __init__