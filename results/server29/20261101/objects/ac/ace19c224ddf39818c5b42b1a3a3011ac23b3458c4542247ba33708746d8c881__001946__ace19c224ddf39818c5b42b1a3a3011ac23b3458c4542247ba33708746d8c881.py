def __create_fn__(_type_artifact_location, _type_description, _type_properties, _type_rectangles, _type_regions, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _dflt_description, _dflt_properties, _dflt_rectangles, _dflt_regions, _return_type):
 def __init__(self,artifact_location:_type_artifact_location,description:_type_description=_dflt_description,properties:_type_properties=_dflt_properties,rectangles:_type_rectangles=_dflt_rectangles,regions:_type_regions=_dflt_regions)->_return_type:
  self.artifact_location=artifact_location
  self.description=description
  self.properties=properties
  self.rectangles=rectangles
  self.regions=regions
 return __init__