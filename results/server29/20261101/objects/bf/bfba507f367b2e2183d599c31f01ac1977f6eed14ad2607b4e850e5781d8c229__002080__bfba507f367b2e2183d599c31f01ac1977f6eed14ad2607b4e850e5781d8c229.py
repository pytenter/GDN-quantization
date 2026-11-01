def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.uri,self.line,self.message,self.start_column,self.end_column,self.snippet,self.function,)==(other.uri,other.line,other.message,other.start_column,other.end_column,other.snippet,other.function,)
  return NotImplemented
 return __eq__