def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.username,self.fullname,self.avatar_url,self.details,self.is_following,self.is_pro,self.num_models,self.num_datasets,self.num_spaces,self.num_discussions,self.num_papers,self.num_upvotes,self.num_likes,self.num_following,self.num_followers,self.orgs,)==(other.username,other.fullname,other.avatar_url,other.details,other.is_following,other.is_pro,other.num_models,other.num_datasets,other.num_spaces,other.num_discussions,other.num_papers,other.num_upvotes,other.num_likes,other.num_following,other.num_followers,other.orgs,)
  return NotImplemented
 return __eq__