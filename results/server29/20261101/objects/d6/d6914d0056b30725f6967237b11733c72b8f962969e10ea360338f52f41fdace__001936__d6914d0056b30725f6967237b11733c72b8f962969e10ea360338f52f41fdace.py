def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.byte_length,self.byte_offset,self.char_length,self.char_offset,self.end_column,self.end_line,self.message,self.properties,self.snippet,self.source_language,self.start_column,self.start_line,)==(other.byte_length,other.byte_offset,other.char_length,other.char_offset,other.end_column,other.end_line,other.message,other.properties,other.snippet,other.source_language,other.start_column,other.start_line,)
  return NotImplemented
 return __eq__