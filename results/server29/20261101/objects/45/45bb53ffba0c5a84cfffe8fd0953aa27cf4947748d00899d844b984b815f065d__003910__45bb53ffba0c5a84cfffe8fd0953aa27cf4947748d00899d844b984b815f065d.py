def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.avatar_url,self.name,self.fullname,self.details,self.is_verified,self.is_following,self.num_users,self.num_models,self.num_spaces,self.num_datasets,self.num_followers,)==(other.avatar_url,other.name,other.fullname,other.details,other.is_verified,other.is_following,other.num_users,other.num_models,other.num_spaces,other.num_datasets,other.num_followers,)
  return NotImplemented
 return __eq__