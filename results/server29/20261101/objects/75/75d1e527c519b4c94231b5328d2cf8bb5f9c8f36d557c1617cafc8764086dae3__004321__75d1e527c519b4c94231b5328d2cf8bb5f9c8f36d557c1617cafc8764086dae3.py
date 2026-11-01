def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.num_image_tokens,self.num_video_tokens,self.num_audio_tokens,self.num_image_patches,)==(other.num_image_tokens,other.num_video_tokens,other.num_audio_tokens,other.num_image_patches,)
  return NotImplemented
 return __eq__