def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.code,self.closure,self.globals,self.locals,self.builtins,self.code_options,self.name_to_modrec,)==(other.code,other.closure,other.globals,other.locals,other.builtins,other.code_options,other.name_to_modrec,)
  return NotImplemented
 return __eq__