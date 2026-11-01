def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.id,self.created_at,self.docker_image,self.space_id,self.command,self.arguments,self.environment,self.secrets,self.flavor,self.status,self.owner,self.endpoint,self.url,)==(other.id,other.created_at,other.docker_image,other.space_id,other.command,other.arguments,other.environment,other.secrets,other.flavor,other.status,other.owner,other.endpoint,other.url,)
  return NotImplemented
 return __eq__