def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.exported_program,self.state_dict,self.constants,self.example_inputs,)==(other.exported_program,other.state_dict,other.constants,other.example_inputs,)
  return NotImplemented
 return __eq__