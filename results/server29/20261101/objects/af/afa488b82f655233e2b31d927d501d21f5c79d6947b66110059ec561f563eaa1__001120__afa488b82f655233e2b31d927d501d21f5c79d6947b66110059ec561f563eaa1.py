def __create_fn__():
 def __hash__(self):
  return hash((self.num_warps,self.num_ctas,self.num_stages,self.num_buffers_warp_spec,self.num_consumer_groups,self.reg_dec_producer,self.reg_inc_consumer,self.maxnreg,self.cluster_dims,self.ptx_version,self.enable_fp_fusion,self.launch_cooperative_grid,self.supported_fp8_dtypes,self.deprecated_fp8_dtypes,self.default_dot_input_precision,self.allowed_dot_input_precisions,self.max_num_imprecise_acc_default,self.extern_libs,self.debug,self.backend_name,self.sanitize_overflow,self.arch,))
 return __hash__