def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.compute_environment,self.distributed_type,self.mixed_precision,self.use_cpu,self.debug,self.ec2_instance_type,self.iam_role_name,self.image_uri,self.profile,self.region,self.num_machines,self.gpu_ids,self.base_job_name,self.pytorch_version,self.transformers_version,self.py_version,self.sagemaker_inputs_file,self.sagemaker_metrics_file,self.additional_args,self.dynamo_config,self.enable_cpu_affinity,)==(other.compute_environment,other.distributed_type,other.mixed_precision,other.use_cpu,other.debug,other.ec2_instance_type,other.iam_role_name,other.image_uri,other.profile,other.region,other.num_machines,other.gpu_ids,other.base_job_name,other.pytorch_version,other.transformers_version,other.py_version,other.sagemaker_inputs_file,other.sagemaker_metrics_file,other.additional_args,other.dynamo_config,other.enable_cpu_affinity,)
  return NotImplemented
 return __eq__