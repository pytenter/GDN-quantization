def __create_fn__(_type_filename, _type_lineno, _type_instruction_pointer, _type_inst, _type_failed, _type_reason, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_failed, _dflt_reason, _return_type):
 def __init__(self,filename:_type_filename,lineno:_type_lineno,instruction_pointer:_type_instruction_pointer,inst:_type_inst,failed:_type_failed=_dflt_failed,reason:_type_reason=_dflt_reason)->_return_type:
  self.filename=filename
  self.lineno=lineno
  self.instruction_pointer=instruction_pointer
  self.inst=inst
  self.failed=failed
  self.reason=reason
 return __init__