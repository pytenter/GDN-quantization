def __create_fn__(_type_frames, _type_message, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_frames, _dflt_message, _return_type):
 def __init__(self,frames:_type_frames=_HAS_DEFAULT_FACTORY,message:_type_message=_dflt_message)->_return_type:
  self.frames=_dflt_frames() if frames is _HAS_DEFAULT_FACTORY else frames
  self.message=message
 return __init__