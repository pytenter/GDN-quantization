def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.name,self.message_default_template,self.short_description,self.full_description,self.full_description_markdown,self.help_uri,)==(other.id,other.name,other.message_default_template,other.short_description,other.full_description,other.full_description_markdown,other.help_uri,)
  return NotImplemented
 return __eq__