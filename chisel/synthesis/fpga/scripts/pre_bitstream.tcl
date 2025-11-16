# Pre-bitstream hook - 在生成比特流前执行
# 降低 DRC 检查的严重性，允许没有引脚约束的设计通过

puts "执行 pre-bitstream hook..."
puts "降低 DRC 检查严重性..."

# 将 DRC 错误降级为警告
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
set_property SEVERITY {Warning} [get_drc_checks RPBF-3]

puts "DRC 检查配置完成"
