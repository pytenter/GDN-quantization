def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(opcode={self.opcode!r}, opname={self.opname!r}, arg={self.arg!r}, argval={self.argval!r}, offset={self.offset!r}, starts_line={self.starts_line!r}, is_jump_target={self.is_jump_target!r}, positions={self.positions!r}, target={self.target!r}, exn_tab_entry={self.exn_tab_entry!r}, argrepr={self.argrepr!r})"
 return __repr__