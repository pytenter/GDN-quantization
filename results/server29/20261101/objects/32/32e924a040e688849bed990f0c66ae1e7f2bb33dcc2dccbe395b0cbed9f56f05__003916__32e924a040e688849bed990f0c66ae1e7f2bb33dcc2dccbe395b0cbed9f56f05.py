def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.authors,self.published_at,self.title,self.summary,self.upvotes,self.discussion_id,self.source,self.comments,self.submitted_at,self.submitted_by,)==(other.id,other.authors,other.published_at,other.title,other.summary,other.upvotes,other.discussion_id,other.source,other.comments,other.submitted_at,other.submitted_by,)
  return NotImplemented
 return __eq__