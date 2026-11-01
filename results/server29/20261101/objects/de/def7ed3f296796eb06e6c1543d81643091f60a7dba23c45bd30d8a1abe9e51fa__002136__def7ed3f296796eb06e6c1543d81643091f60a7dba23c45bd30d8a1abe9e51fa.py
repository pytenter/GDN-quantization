def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.py_obj,self.name,self.filename,self.code,)==(other.py_obj,other.name,other.filename,other.code,)
  return NotImplemented
 return __eq__