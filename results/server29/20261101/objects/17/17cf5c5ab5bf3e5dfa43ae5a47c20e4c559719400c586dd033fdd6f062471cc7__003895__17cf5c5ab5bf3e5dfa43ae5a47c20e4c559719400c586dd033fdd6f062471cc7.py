def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.slug,self.title,self.owner,self.items,self.last_updated,self.position,self.private,self.theme,self.upvotes,self.description,)==(other.slug,other.title,other.owner,other.items,other.last_updated,other.position,other.private,other.theme,other.upvotes,other.description,)
  return NotImplemented
 return __eq__