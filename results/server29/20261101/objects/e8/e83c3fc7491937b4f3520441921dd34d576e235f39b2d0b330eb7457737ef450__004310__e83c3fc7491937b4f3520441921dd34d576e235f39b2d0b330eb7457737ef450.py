def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.total_num_frames,self.fps,self.width,self.height,self.duration,self.video_backend,self.frames_indices,)==(other.total_num_frames,other.fps,other.width,other.height,other.duration,other.video_backend,other.frames_indices,)
  return NotImplemented
 return __eq__