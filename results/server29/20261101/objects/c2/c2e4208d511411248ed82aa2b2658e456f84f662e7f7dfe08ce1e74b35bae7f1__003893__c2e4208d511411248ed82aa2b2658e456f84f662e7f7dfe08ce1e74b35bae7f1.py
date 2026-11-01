def __create_fn__(_type_slug, _type_title, _type_owner, _type_items, _type_last_updated, _type_position, _type_private, _type_theme, _type_upvotes, _type_description, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_description, _return_type):
 def __init__(self,slug:_type_slug,title:_type_title,owner:_type_owner,items:_type_items,last_updated:_type_last_updated,position:_type_position,private:_type_private,theme:_type_theme,upvotes:_type_upvotes,description:_type_description=_dflt_description)->_return_type:
  self.slug=slug
  self.title=title
  self.owner=owner
  self.items=items
  self.last_updated=last_updated
  self.position=position
  self.private=private
  self.theme=theme
  self.upvotes=upvotes
  self.description=description
 return __init__