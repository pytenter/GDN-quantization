def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.code,self.closure,self.globals,self.locals,self.builtins,self.code_options,)==(other.code,other.closure,other.globals,other.locals,other.builtins,other.code_options,)
  return NotImplemented
 return __eq__