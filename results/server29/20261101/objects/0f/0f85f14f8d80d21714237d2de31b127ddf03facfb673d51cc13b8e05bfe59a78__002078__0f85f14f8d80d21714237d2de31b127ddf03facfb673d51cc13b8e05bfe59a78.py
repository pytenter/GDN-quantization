def __create_fn__(_type_uri, _type_line, _type_message, _type_start_column, _type_end_column, _type_snippet, _type_function, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_uri, _dflt_line, _dflt_message, _dflt_start_column, _dflt_end_column, _dflt_snippet, _dflt_function, _return_type):
 def __init__(self,uri:_type_uri=_dflt_uri,line:_type_line=_dflt_line,message:_type_message=_dflt_message,start_column:_type_start_column=_dflt_start_column,end_column:_type_end_column=_dflt_end_column,snippet:_type_snippet=_dflt_snippet,function:_type_function=_dflt_function)->_return_type:
  self.uri=uri
  self.line=line
  self.message=message
  self.start_column=start_column
  self.end_column=end_column
  self.snippet=snippet
  self.function=function
 return __init__