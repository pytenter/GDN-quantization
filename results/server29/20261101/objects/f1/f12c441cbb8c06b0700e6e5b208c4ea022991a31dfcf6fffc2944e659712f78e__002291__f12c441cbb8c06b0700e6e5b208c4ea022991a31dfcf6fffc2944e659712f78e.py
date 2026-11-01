def __create_fn__(_type_table_name, _type_column_names, _type_num_rows_added, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_num_rows_added, _return_type):
 def __init__(self,table_name:_type_table_name,column_names:_type_column_names,num_rows_added:_type_num_rows_added=_dflt_num_rows_added)->_return_type:
  self.table_name=table_name
  self.column_names=column_names
  self.num_rows_added=num_rows_added
 return __init__