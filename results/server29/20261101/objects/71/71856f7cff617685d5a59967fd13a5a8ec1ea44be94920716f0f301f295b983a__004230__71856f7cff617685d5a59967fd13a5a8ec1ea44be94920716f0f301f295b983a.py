def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.height,self.width,self.longest_edge,self.shortest_edge,self.max_height,self.max_width,)==(other.height,other.width,other.longest_edge,other.shortest_edge,other.max_height,other.max_width,)
  return NotImplemented
 return __eq__