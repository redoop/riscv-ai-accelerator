# FPGA 构建指南

本文档详细说明如何在 AWS F1 实例上构建 RISC-V AI 加速器 FPGA 镜像。

## 前置条件

### 1. AWS 环境

- AWS 账户（已配置 F1 实例访问权限）
- F1 实例（推荐 f1.2xlarge 用于开发）
- FPGA Developer AMI（预装 Vivado）

### 2. 本地环境

- Git
- AWS CLI
- SSH 客户端

## 构建流程

### 步骤 1：准备 RTL 代码

在本地机器上生成 Verilog：

```bash
cd chisel
sbt "runMain edgeai.SimpleEdgeAiSoCMain"
```

生成的文件位于：`chisel/generated/simple_edgeaisoc/`

### 步骤 2：上传到 F1 实例

```bash
# 打包项目
tar czf fpga_project.tar.gz chisel/synthesis/fpga chisel/generated

# 上传到 F1 实例
scp -i your-key.pem fpga_project.tar.gz centos@<f1-ip>:~/

# 在 F1 实例上解压
ssh -i your-key.pem centos@<f1-ip>
tar xzf fpga_project.tar.gz
```

### 步骤 3：设置 AWS FPGA 环境

```bash
# 克隆 AWS FPGA 仓库（如果未安装）
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh
source hdk_setup.sh
```

### 步骤 4：运行 Vivado 构建

```bash
cd ~/chisel/synthesis/fpga

# 方法 1：使用自动化脚本
vivado -mode batch -source scripts/build_fpga.tcl

# 方法 2：交互式构建
vivado &
# 在 Vivado GUI 中：
# - File -> Open Project
# - 选择 build/fpga_project.xpr
# - Flow -> Run Implementation
# - Flow -> Generate Bitstream
```

构建时间：约 2-4 小时

### 步骤 5：检查构建结果

```bash
# 检查时序
cat build/reports/timing_summary.rpt | grep WNS

# 检查资源利用率
cat build/reports/utilization.rpt

# 检查功耗
cat build/reports/power.rpt
```

**通过标准：**
- WNS (Worst Negative Slack) > 0
- 资源利用率 < 80%
- 无严重警告

### 步骤 6：生成 DCP 文件

构建成功后，DCP 文件位于：
```
build/checkpoints/to_aws/SH_CL_routed.dcp
```

这个文件将用于创建 AWS AFI。

## 常见问题

### Q1: 时序违例 (WNS < 0)

**解决方案：**
1. 检查关键路径：`build/reports/timing_summary.rpt`
2. 添加流水线寄存器
3. 降低目标频率（修改 `constraints/timing.xdc`）
4. 优化综合策略

### Q2: 资源不足

**解决方案：**
1. 检查资源报告：`build/reports/utilization.rpt`
2. 移除调试逻辑（ILA）
3. 使用更大的 FPGA（f1.16xlarge）
4. 简化设计

### Q3: 构建失败

**解决方案：**
1. 检查日志：`build/vivado.log`
2. 验证 RTL 语法
3. 检查约束文件
4. 确认 Vivado 版本兼容性

## 优化建议

### 时序优化

```tcl
# 在 build_fpga.tcl 中添加
set_property STRATEGY Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
```

### 资源优化

```tcl
# 减少 BRAM 使用
set_property RAM_STYLE distributed [get_cells -hier -filter {RTL_RAM_TYPE}]
```

### 功耗优化

```tcl
# 启用功耗优化
set_property STRATEGY Power_DefaultOpt [get_runs impl_1]
```

## 下一步

构建完成后，继续 [创建 AFI](../aws-deployment/create_afi.sh)。
