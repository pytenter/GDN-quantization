def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.author,self.sha,self.created_at,self.last_modified,self.private,self.gated,self.disabled,self.downloads,self.downloads_all_time,self.likes,self.paperswithcode_id,self.tags,self.trending_score,self.card_data,self.siblings,self.xet_enabled,)==(other.id,other.author,other.sha,other.created_at,other.last_modified,other.private,other.gated,other.disabled,other.downloads,other.downloads_all_time,other.likes,other.paperswithcode_id,other.tags,other.trending_score,other.card_data,other.siblings,other.xet_enabled,)
  return NotImplemented
 return __eq__