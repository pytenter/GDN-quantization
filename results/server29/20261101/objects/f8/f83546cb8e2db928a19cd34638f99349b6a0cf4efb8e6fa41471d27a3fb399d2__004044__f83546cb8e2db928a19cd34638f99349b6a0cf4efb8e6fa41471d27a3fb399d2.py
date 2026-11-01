def __create_fn__():
 def __eq__(self,other):
  if other.__class__ is self.__class__:
   return (self.load_in_8bit,self.llm_int8_threshold,self.load_in_4bit,self.bnb_4bit_quant_type,self.bnb_4bit_use_double_quant,self.bnb_4bit_compute_dtype,self.torch_dtype,self.skip_modules,self.keep_in_fp32_modules,)==(other.load_in_8bit,other.llm_int8_threshold,other.load_in_4bit,other.bnb_4bit_quant_type,other.bnb_4bit_use_double_quant,other.bnb_4bit_compute_dtype,other.torch_dtype,other.skip_modules,other.keep_in_fp32_modules,)
  return NotImplemented
 return __eq__