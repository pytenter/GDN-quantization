def __create_fn__(_type_id, _type_created_at, _type_job_spec, _type_schedule, _type_suspend, _type_concurrency, _type_status, _type_owner, MISSING, _HAS_DEFAULT_FACTORY, __dataclass_builtins_object__, _return_type):
 def __init__(self,id:_type_id,created_at:_type_created_at,job_spec:_type_job_spec,schedule:_type_schedule,suspend:_type_suspend,concurrency:_type_concurrency,status:_type_status,owner:_type_owner)->_return_type:
  self.id=id
  self.created_at=created_at
  self.job_spec=job_spec
  self.schedule=schedule
  self.suspend=suspend
  self.concurrency=concurrency
  self.status=status
  self.owner=owner
 return __init__