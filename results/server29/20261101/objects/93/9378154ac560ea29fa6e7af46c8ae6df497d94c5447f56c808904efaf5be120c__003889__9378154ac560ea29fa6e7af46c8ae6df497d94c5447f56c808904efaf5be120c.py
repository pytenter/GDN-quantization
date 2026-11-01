def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.author,self.sha,self.created_at,self.last_modified,self.private,self.gated,self.disabled,self.host,self.subdomain,self.likes,self.sdk,self.tags,self.siblings,self.trending_score,self.card_data,self.runtime,self.models,self.datasets,self.xet_enabled,)==(other.id,other.author,other.sha,other.created_at,other.last_modified,other.private,other.gated,other.disabled,other.host,other.subdomain,other.likes,other.sdk,other.tags,other.siblings,other.trending_score,other.card_data,other.runtime,other.models,other.datasets,other.xet_enabled,)
  return NotImplemented
 return __eq__