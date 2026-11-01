def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.name,self.fn_call_name,self.args,self.ret,self.sub_idx,self.is_pure,)==(other.name,other.fn_call_name,other.args,other.ret,other.sub_idx,other.is_pure,)
  return NotImplemented
 return __eq__