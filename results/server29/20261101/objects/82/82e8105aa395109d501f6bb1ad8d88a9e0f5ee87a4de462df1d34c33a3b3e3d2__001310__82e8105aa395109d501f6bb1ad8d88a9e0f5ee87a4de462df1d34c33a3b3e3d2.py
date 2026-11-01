def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.opcode,self.opname,self.arg,self.argval,self.offset,self.starts_line,self.is_jump_target,self.positions,self.target,self.exn_tab_entry,self.argrepr,)==(other.opcode,other.opname,other.arg,other.argval,other.offset,other.starts_line,other.is_jump_target,other.positions,other.target,other.exn_tab_entry,other.argrepr,)
  return NotImplemented
 return __eq__