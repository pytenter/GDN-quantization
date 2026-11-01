def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.hf_ds_config,self.gradient_accumulation_steps,self.gradient_clipping,self.zero_stage,self.is_train_batch_min,self.offload_optimizer_device,self.offload_param_device,self.offload_optimizer_nvme_path,self.offload_param_nvme_path,self.zero3_init_flag,self.zero3_save_16bit_model,self.transformer_moe_cls_names,self.enable_msamp,self.msamp_opt_level,)==(other.hf_ds_config,other.gradient_accumulation_steps,other.gradient_clipping,other.zero_stage,other.is_train_batch_min,other.offload_optimizer_device,other.offload_param_device,other.offload_optimizer_nvme_path,other.offload_param_nvme_path,other.zero3_init_flag,other.zero3_save_16bit_model,other.transformer_moe_cls_names,other.enable_msamp,other.msamp_opt_level,)
  return NotImplemented
 return __eq__