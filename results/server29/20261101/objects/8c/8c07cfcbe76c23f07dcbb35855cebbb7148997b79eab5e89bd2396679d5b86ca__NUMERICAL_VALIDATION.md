# dP32 数值与生命周期验证

[]

沿用原0.5%跨 replay、0.1%同 actual-state 读出的工程门禁，不改阈值。覆盖31/32/33、63/64/65、prompt_end、未满窗口preserve、强弱衰减、exponent、实际32 heads及257步弱衰减；原跨轨迹局部标签保留，同窗口base反事实单独解释继承分叉。验证使用dense reference，吞吐路径的capture_validation=False。验证失败后没有启动性能测试。原代码和旧正式输出均未修改。
