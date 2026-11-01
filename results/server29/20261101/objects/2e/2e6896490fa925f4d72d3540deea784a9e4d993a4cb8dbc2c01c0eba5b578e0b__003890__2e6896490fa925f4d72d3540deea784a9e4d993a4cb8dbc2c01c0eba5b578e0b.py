def __create_fn__(_type_item_object_id, _type_item_id, _type_item_type, _type_position, _type_note, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_note, _return_type):
 def __init__(self,item_object_id:_type_item_object_id,item_id:_type_item_id,item_type:_type_item_type,position:_type_position,note:_type_note=_dflt_note)->_return_type:
  self.item_object_id=item_object_id
  self.item_id=item_id
  self.item_type=item_type
  self.position=position
  self.note=note
 return __init__