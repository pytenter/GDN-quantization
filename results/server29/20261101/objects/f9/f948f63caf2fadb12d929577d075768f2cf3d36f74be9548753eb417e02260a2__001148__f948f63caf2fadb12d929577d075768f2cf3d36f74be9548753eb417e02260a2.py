def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.ncalls,self.tottime,self.percall_tottime,self.cumtime,self.percall_cumtime,self.file_name,self.line_number,)==(other.ncalls,other.tottime,other.percall_tottime,other.cumtime,other.percall_cumtime,other.file_name,other.line_number,)
  return NotImplemented
 return __eq__