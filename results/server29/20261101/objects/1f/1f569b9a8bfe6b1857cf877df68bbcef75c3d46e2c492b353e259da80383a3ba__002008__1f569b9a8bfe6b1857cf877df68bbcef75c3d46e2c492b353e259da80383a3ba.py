def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.default_configuration,self.deprecated_guids,self.deprecated_ids,self.deprecated_names,self.full_description,self.guid,self.help,self.help_uri,self.message_strings,self.name,self.properties,self.relationships,self.short_description,)==(other.id,other.default_configuration,other.deprecated_guids,other.deprecated_ids,other.deprecated_names,other.full_description,other.guid,other.help,other.help_uri,other.message_strings,other.name,other.properties,other.relationships,other.short_description,)
  return NotImplemented
 return __eq__