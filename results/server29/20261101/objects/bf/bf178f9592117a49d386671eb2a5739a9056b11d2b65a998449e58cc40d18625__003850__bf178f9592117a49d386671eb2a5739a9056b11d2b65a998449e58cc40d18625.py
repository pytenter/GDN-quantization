def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.safe,self.status,self.av_scan,self.pickle_import_scan,)==(other.safe,other.status,other.av_scan,other.pickle_import_scan,)
  return NotImplemented
 return __eq__