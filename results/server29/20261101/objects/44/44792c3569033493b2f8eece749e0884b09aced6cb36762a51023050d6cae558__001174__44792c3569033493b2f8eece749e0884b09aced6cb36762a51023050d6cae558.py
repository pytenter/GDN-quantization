def __create_fn__():
 def __hash__(self):
  return hash((self.lower,self.upper,self.is_bool,self.is_int,self.is_float,))
 return __hash__