def __create_fn__():
 def __repr__(self):
  return self.__class__.__qualname__ + f"(filename={self.filename!r}, lineno={self.lineno!r}, instruction_pointer={self.instruction_pointer!r}, inst={self.inst!r}, failed={self.failed!r}, reason={self.reason!r})"
 return __repr__