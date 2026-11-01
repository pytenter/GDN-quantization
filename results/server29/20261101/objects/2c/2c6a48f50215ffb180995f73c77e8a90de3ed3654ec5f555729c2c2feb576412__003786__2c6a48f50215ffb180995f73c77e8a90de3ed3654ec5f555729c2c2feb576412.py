def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.namespace,self.repository,self.status,self.health_route,self.url,self.framework,self.revision,self.task,self.created_at,self.updated_at,self.type,self.raw,)==(other.name,other.namespace,other.repository,other.status,other.health_route,other.url,other.framework,other.revision,other.task,other.created_at,other.updated_at,other.type,other.raw,)
  return NotImplemented
 return __eq__