def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.filename,self.lineno,self.instruction_pointer,self.inst,self.failed,self.reason,)==(other.filename,other.lineno,other.instruction_pointer,other.inst,other.failed,other.reason,)
  return NotImplemented
 return __eq__