def __create_fn__(_type_opcode, _type_opname, _type_arg, _type_argval, _type_offset, _type_starts_line, _type_is_jump_target, _type_positions, _type_target, _type_exn_tab_entry, _type_argrepr, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_offset, _dflt_starts_line, _dflt_is_jump_target, _dflt_positions, _dflt_target, _dflt_exn_tab_entry, _dflt_argrepr, _return_type):
 def __init__(self,opcode:_type_opcode,opname:_type_opname,arg:_type_arg,argval:_type_argval,offset:_type_offset=_dflt_offset,starts_line:_type_starts_line=_dflt_starts_line,is_jump_target:_type_is_jump_target=_dflt_is_jump_target,positions:_type_positions=_dflt_positions,target:_type_target=_dflt_target,exn_tab_entry:_type_exn_tab_entry=_dflt_exn_tab_entry,argrepr:_type_argrepr=_dflt_argrepr)->_return_type:
  self.opcode=opcode
  self.opname=opname
  self.arg=arg
  self.argval=argval
  self.offset=offset
  self.starts_line=starts_line
  self.is_jump_target=is_jump_target
  self.positions=positions
  self.target=target
  self.exn_tab_entry=exn_tab_entry
  self.argrepr=argrepr
 return __init__