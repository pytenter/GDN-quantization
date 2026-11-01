def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.table_name,self.column_names,self.num_rows_added,)==(other.table_name,other.column_names,other.num_rows_added,)
  return NotImplemented
 return __eq__