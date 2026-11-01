def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.docker_image,self.space_id,self.command,self.arguments,self.environment,self.secrets,self.flavor,self.timeout,self.tags,self.arch,)==(other.docker_image,other.space_id,other.command,other.arguments,other.environment,other.secrets,other.flavor,other.timeout,other.tags,other.arch,)
  return NotImplemented
 return __eq__