def __create_fn__(_type_total_num_frames, _type_fps, _type_width, _type_height, _type_duration, _type_video_backend, _type_frames_indices, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_fps, _dflt_width, _dflt_height, _dflt_duration, _dflt_video_backend, _dflt_frames_indices, _return_type):
 def __init__(self,total_num_frames:_type_total_num_frames,fps:_type_fps=_dflt_fps,width:_type_width=_dflt_width,height:_type_height=_dflt_height,duration:_type_duration=_dflt_duration,video_backend:_type_video_backend=_dflt_video_backend,frames_indices:_type_frames_indices=_dflt_frames_indices)->_return_type:
  self.total_num_frames=total_num_frames
  self.fps=fps
  self.width=width
  self.height=height
  self.duration=duration
  self.video_backend=video_backend
  self.frames_indices=frames_indices
 return __init__