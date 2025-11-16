# RISC-V AI 加速器 AWS FPGA 验证方案

**版本：** 1.0  
**日期：** 2025年11月16日  
**目标平台：** AWS F1 实例 (Xilinx UltraScale+ VU9P FPGA)

---

## 📋 执行检查清单

### 阶段 1：本地准备（已完成 ✅）

- [x] **环境检查**
  - [x] Java 11+ 已安装
  - [x] sbt 已安装
  - [x] AWS CLI 已安装（v2.31.37）
  - [x] Git 已配置

- [x] **RTL 设计**
  - [x] Chisel 代码编写完成
  - [x] Verilog 生成成功（3765 行）
  - [x] 本地测试全部通过（20/20）

- [x] **FPGA 适配层**
  - [x] fpga_top.v 顶层封装
  - [x] clock_gen.v 时钟生成
  - [x] io_adapter.v IO 适配
  - [x] 约束文件（timing.xdc, pins.xdc, physical.xdc）

- [x] **测试脚本**
  - [x] 功能测试脚本（test_*.sh）
  - [x] 性能测试脚本（benchmark_*.sh）
  - [x] 测试向量准备

- [x] **文档准备**
  - [x] QUICK_START.md
  - [x] LOCAL_TEST_GUIDE.md
  - [x] SETUP_GUIDE.md
  - [x] BUILD_GUIDE.md
  - [x] TEST_GUIDE.md
  - [x] TEST_RESULTS.md

### 阶段 2：AWS 环境配置（100% 完成 ✅）

- [x] **AWS 账户准备**
  - [x] AWS 账户已创建
  - [x] IAM 权限已配置（EC2, S3, FPGA）
  - [x] F1 实例配额已申请（96 个）
  - [x] 密钥对已创建（fpga-dev-key）

- [x] **AMI 订阅**
  - [x] FPGA Developer AMI 已订阅（ami-0cb1b6ae2ff99f8bf）
  - [x] 网络配置完成（VPC, Subnet）
  - [x] 安全组已创建（sg-03d27449f82b54360）

- [x] **启动 F2 实例**（已完成 ✅）
  - [x] 订阅 Vivado AMI (ami-0b359c50bdba2aac0)
  - [x] 启动 f2.6xlarge Spot 实例
  - [x] 配置安全组和网络
  - [x] SSH 连接测试成功
  
  **实例信息**：
  - 实例 ID: i-00d976d528e721c43
  - 公网 IP: 54.81.161.62
  - 用户名: ubuntu
  - Spot 请求 ID: sir-nh5zbwyn
  - 详见：`aws-deployment/F2_VIVADO_GUIDE.md`

- [x] **Vivado 环境验证**（已完成 ✅）
  - [x] Vivado 2025.1 已确认可用
  - [x] Vivado 路径：`/tools/Xilinx/2025.1/Vivado/bin/vivado`
  - [x] 环境设置脚本已创建
  - [x] 使用指南已创建

- [x] **项目部署**（已完成 ✅）
  - [x] 上传项目代码到 F2（44KB）
  - [x] 设置 Vivado 环境
  - [x] 验证环境配置
  - [x] 修复端口不匹配问题
  - [x] Vivado 构建已启动（进程 PID: 7040）
  
  **构建状态**：
  - 开始时间: 2025-11-16 09:14 UTC (17:14 北京时间)
  - 综合完成: 2025-11-16 09:17 UTC (17:17 北京时间)
  - 当前阶段: 实现 (Implementation) 进行中
  - 预计完成: 2025-11-16 10:17-11:17 UTC (18:17-19:17 北京时间)
  - 详见：`aws-deployment/BUILD_PROGRESS.md`

### 阶段 3：FPGA 构建（40% 完成 🔄）

- [x] **Vivado 综合**（已完成 ✅）
  - [x] 运行 build_fpga_f2.tcl
  - [x] 综合完成（用时 9 分钟）
  - [x] 检查综合报告
  - [x] 资源利用率检查
  
  **综合结果**：
  - 完成时间: 2025-11-16 09:17 UTC (17:17 北京时间)
  - RAM 资源: CompactAccel (256x32), BitNetAccel (256x32)
  - DSP 资源: 4 个 DSP48E2 (SimpleCompactAccel)
  - 优化: 43 个未使用寄存器被移除
  - 报告: `fpga-project/build/reports/utilization_synth.rpt`

- [ ] **Vivado 实现**（进行中 🔄）
  - [x] 启动实现流程
  - [ ] 优化 (Opt Design) - 进行中
  - [ ] 布局 (Place Design) - 待执行
  - [ ] 布线 (Route Design) - 待执行
  - [ ] 实现完成（预计 1-2 小时）
  
  **当前状态**：
  - 进程 PID: 7040
  - 阶段: Implementation (opt_design)
  - 开始时间: 2025-11-16 09:17 UTC
  - 日志: `fpga-project/build/logs/vivado_build.log`

- [ ] **时序分析**（待执行 ⏳）
  - [ ] WNS > 0（无时序违例）
  - [ ] 工作频率达到 100MHz
  - [ ] 关键路径分析
  - [ ] 时序报告归档

- [ ] **生成 DCP**（待执行 ⏳）
  - [ ] DCP 文件生成成功
  - [ ] 文件位置：build/checkpoints/to_aws/SH_CL_routed.dcp
  - [ ] 文件大小检查

### 阶段 4：AFI 创建（待执行 ⏳）

- [ ] **S3 准备**
  - [ ] 创建 S3 bucket
  - [ ] 配置 bucket 权限
  - [ ] 上传 DCP 文件

- [ ] **创建 AFI**
  - [ ] 运行 create_afi.sh
  - [ ] 获取 AFI ID
  - [ ] 获取 AGFI ID
  - [ ] 保存 AFI 信息

- [ ] **等待 AFI 可用**
  - [ ] 监控 AFI 状态
  - [ ] 状态变为 "available"（预计 30-60 分钟）
  - [ ] 检查 S3 日志（如有错误）

### 阶段 5：FPGA 部署与测试（待执行 ⏳）

- [ ] **加载 AFI**
  - [ ] 清除现有 FPGA 镜像
  - [ ] 加载新 AFI
  - [ ] 验证加载状态

- [ ] **功能测试**
  - [ ] 处理器启动测试
  - [ ] UART 通信测试
  - [ ] GPIO 读写测试
  - [ ] CompactAccel 2x2 测试
  - [ ] CompactAccel 4x4 测试
  - [ ] CompactAccel 8x8 测试
  - [ ] BitNetAccel 2x2 测试
  - [ ] BitNetAccel 8x8 测试
  - [ ] 中断处理测试

- [ ] **性能测试**
  - [ ] GOPS 测量（目标 6.4）
  - [ ] 延迟测量（目标 <100 cycles）
  - [ ] 吞吐量测试（目标 >90%）
  - [ ] 连续运行稳定性测试

- [ ] **功耗测试**
  - [ ] 读取 FPGA 功耗
  - [ ] 估算 ASIC 功耗
  - [ ] 功耗报告生成

### 阶段 6：文档与交付（待执行 ⏳）

- [ ] **测试报告**
  - [ ] 功能测试报告
  - [ ] 性能测试报告
  - [ ] 问题与解决方案

- [ ] **技术文档**
  - [ ] 综合报告
  - [ ] 时序报告
  - [ ] 功耗报告
  - [ ] 资源利用报告

- [ ] **代码归档**
  - [ ] 最终 Verilog 代码
  - [ ] 约束文件
  - [ ] 测试脚本
  - [ ] AFI 信息

---

## 一、方案概述

### 1.1 验证目标

本方案使用 AWS F1 FPGA 实例对 RISC-V AI 加速器芯片进行原型验证，主要验证：

- **功能正确性**：验证 PicoRV32 处理器、CompactAccel 和 BitNetAccel 加速器的功能
- **时序性能**：验证设计在 100MHz 工作频率下的时序收敛
- **系统集成**：验证 UART、GPIO、中断控制等外设的集成
- **性能指标**：测量实际 GOPS 性能，验证 6.4 GOPS @ 100MHz 目标
- **功耗评估**：通过 FPGA 功耗报告估算 ASIC 功耗

### 1.2 AWS F1 实例优势

- **高性能 FPGA**：Xilinx UltraScale+ VU9P，2.5M 逻辑单元，充足资源
- **按需使用**：按小时计费，无需购买昂贵硬件
- **完整工具链**：预装 Vivado 开发套件
- **PCIe 接口**：支持高速数据传输和调试
- **弹性扩展**：可根据需要启动多个实例并行测试

### 1.3 资源需求估算

基于设计规模（73829 标准单元）：

| 资源类型 | 预估使用量 | VU9P 可用量 | 利用率 |
|---------|-----------|------------|--------|
| LUT | ~50,000 | 1,182,240 | ~4% |
| FF | ~40,000 | 2,364,480 | ~2% |
| BRAM | ~20 | 2,160 | ~1% |
| DSP | 0 (BitNet 无乘法器) | 6,840 | 0% |

资源充足，可支持完整设计实现。

---

## 二、实施方案

### 2.1 开发流程

```
RTL 设计 (Chisel) 
    ↓
生成 Verilog
    ↓
FPGA 约束文件
    ↓
Vivado 综合与实现
    ↓
生成比特流
    ↓
AWS AFI (FPGA Image)
    ↓
F1 实例部署
    ↓
功能与性能测试
```

### 2.2 目录结构

```
chisel/synthesis/fpga/
├── AWS_FPGA_PLAN.md          # 本文档
├── README.md                  # 快速开始指南
├── constraints/               # 约束文件
│   ├── timing.xdc            # 时序约束
│   ├── pins.xdc              # 引脚约束
│   └── physical.xdc          # 物理约束
├── scripts/                   # 自动化脚本
│   ├── setup_aws.sh          # AWS 环境配置
│   ├── build_fpga.tcl        # Vivado 构建脚本
│   ├── create_afi.sh         # 创建 AFI 镜像
│   └── run_tests.sh          # 测试脚本
├── src/                       # FPGA 特定源码
│   ├── fpga_top.v            # FPGA 顶层封装
│   ├── clock_gen.v           # 时钟生成模块
│   └── io_adapter.v          # IO 适配器
├── testbench/                 # 测试平台
│   ├── tb_fpga_top.sv        # 顶层测试
│   └── test_vectors/         # 测试向量
└── docs/                      # 文档
    ├── SETUP_GUIDE.md        # 环境搭建指南
    ├── BUILD_GUIDE.md        # 构建指南
    └── TEST_GUIDE.md         # 测试指南
```

---

## 三、技术方案

### 3.1 时钟方案

AWS F1 提供多个时钟域，设计使用：

- **主时钟**：100 MHz（从 250 MHz PCIe 时钟分频）
- **UART 时钟**：50 MHz（用于波特率生成）
- **调试时钟**：125 MHz（用于 ILA 逻辑分析）

时钟生成使用 Xilinx MMCM/PLL IP 核。

### 3.2 复位方案

- **上电复位**：FPGA 配置完成后自动复位
- **软件复位**：通过 PCIe 寄存器控制
- **复位同步**：使用双触发器同步到各时钟域

### 3.3 IO 映射

| SoC 信号 | FPGA 映射 | 说明 |
|---------|----------|------|
| clock | MMCM 输出 | 100 MHz 系统时钟 |
| reset | PCIe 寄存器 | 软件可控复位 |
| io_uart_tx | PCIe BAR 或 USB-UART | UART 发送 |
| io_uart_rx | PCIe BAR 或 USB-UART | UART 接收 |
| io_gpio_out[31:0] | PCIe BAR 寄存器 | GPIO 输出 |
| io_gpio_in[31:0] | PCIe BAR 寄存器 | GPIO 输入 |
| io_gpio_oe[31:0] | PCIe BAR 寄存器 | GPIO 方向控制 |

### 3.4 调试方案

使用 Xilinx ILA (Integrated Logic Analyzer) 进行在线调试：

- **触发信号**：UART 传输、加速器启动、中断事件
- **采样深度**：8K 样本
- **监控信号**：
  - 处理器状态机
  - 加速器输入/输出
  - 内存访问
  - 中断信号

---

## 四、实施步骤

### 4.1 环境准备

#### 4.1.1 AWS 账户配置

```bash
# 安装 AWS CLI
pip install awscli

# 配置 AWS 凭证
aws configure
# 输入 Access Key ID、Secret Access Key、区域（us-east-1 或 us-west-2）

# 订阅 FPGA Developer AMI
# 在 AWS Marketplace 搜索 "FPGA Developer AMI"
```

#### 4.1.2 启动 F1 实例

```bash
# 启动 f1.2xlarge 实例（开发用）
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type f1.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx

# 连接到实例
ssh -i your-key.pem centos@<instance-ip>
```

#### 4.1.3 安装开发工具

```bash
# 克隆 AWS FPGA 仓库
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh

# 安装 Vivado（如果未预装）
# 下载 Xilinx Vivado 2021.2 或更高版本
```

### 4.2 设计准备

#### 4.2.1 生成 Verilog

```bash
cd chisel
# 生成 Verilog 代码
sbt "runMain edgeai.SimpleEdgeAiSoCMain"

# 输出位置：chisel/generated/simple_edgeaisoc/
```

#### 4.2.2 创建 FPGA 顶层

将 SoC 设计封装为 AWS Shell 兼容的顶层模块（见 `src/fpga_top.v`）。

### 4.3 综合与实现

#### 4.3.1 运行 Vivado 构建

```bash
cd chisel/synthesis/fpga
./scripts/build_fpga.tcl

# 或手动运行
vivado -mode batch -source scripts/build_fpga.tcl
```

构建流程：
1. 读取 RTL 源码
2. 应用约束文件
3. 综合（Synthesis）
4. 实现（Implementation）
5. 生成比特流（Bitstream）

预计构建时间：2-4 小时

#### 4.3.2 时序分析

检查时序报告：
```bash
# 查看 WNS (Worst Negative Slack)
grep "WNS" build/reports/timing_summary.rpt

# 目标：WNS > 0（无时序违例）
```

### 4.4 创建 AFI

#### 4.4.1 生成 DCP 文件

```bash
# Vivado 生成 Design Checkpoint
# 输出：build/checkpoints/to_aws/SH_CL_routed.dcp
```

#### 4.4.2 上传并创建 AFI

```bash
cd chisel/synthesis/fpga
./scripts/create_afi.sh

# 脚本执行：
# 1. 创建 S3 bucket（如果不存在）
# 2. 上传 DCP 到 S3
# 3. 调用 aws ec2 create-fpga-image
# 4. 等待 AFI 生成（约 30-60 分钟）
```

#### 4.4.3 检查 AFI 状态

```bash
aws ec2 describe-fpga-images --fpga-image-ids afi-xxxxxxxxx

# 等待状态变为 "available"
```

### 4.5 部署与测试

#### 4.5.1 加载 AFI

```bash
# 在 F1 实例上
sudo fpga-load-local-image -S 0 -I afi-xxxxxxxxx

# 验证加载
sudo fpga-describe-local-image -S 0 -H
```

#### 4.5.2 运行功能测试

```bash
cd chisel/synthesis/fpga
./scripts/run_tests.sh

# 测试内容：
# 1. 处理器启动测试
# 2. UART 通信测试
# 3. GPIO 读写测试
# 4. CompactAccel 矩阵乘法测试
# 5. BitNetAccel 矩阵乘法测试
# 6. 中断处理测试
# 7. 性能基准测试
```

#### 4.5.3 性能测试

```bash
# 测量 GOPS 性能
./scripts/benchmark_gops.sh

# 预期结果：
# - CompactAccel: ~6.4 GOPS @ 100MHz
# - BitNetAccel: ~6.4 GOPS @ 100MHz（无乘法器）
# - 延迟：<100 cycles（8x8 矩阵）
```

---

## 五、测试计划

### 5.1 功能测试

| 测试项 | 测试方法 | 通过标准 |
|-------|---------|---------|
| 处理器启动 | 加载 bootloader，检查 UART 输出 | 输出 "Boot OK" |
| UART 通信 | 发送/接收测试字符串 | 数据一致 |
| GPIO 读写 | 写入模式，读回验证 | 读写一致 |
| CompactAccel | 2x2, 4x4, 8x8 矩阵乘法 | 结果正确 |
| BitNetAccel | 2x2, 8x8 矩阵乘法 | 结果正确 |
| 中断处理 | 触发中断，检查响应 | 中断正确处理 |
| 内存访问 | 读写不同地址 | 数据正确 |

### 5.2 性能测试

| 指标 | 测试方法 | 目标值 |
|-----|---------|--------|
| 工作频率 | 时序报告 | 100 MHz |
| GOPS | 矩阵运算计时 | 6.4 GOPS |
| 延迟 | ILA 测量 | <100 cycles |
| 吞吐量 | 连续运算 | >90% 利用率 |

### 5.3 功耗测试

```bash
# 读取 FPGA 功耗
sudo fpga-describe-local-image -S 0 -M

# 分析功耗报告
# 估算 ASIC 功耗 = FPGA 功耗 × 缩放因子（约 0.1-0.2）
```

---

## 六、成本估算

### 6.1 AWS F1 实例费用

| 实例类型 | 价格（美东） | 用途 | 预计时长 |
|---------|------------|------|---------|
| f1.2xlarge | $1.65/小时 | 开发测试 | 40 小时 |
| f1.4xlarge | $3.30/小时 | 性能测试 | 10 小时 |

**总计**：约 $99（开发）+ $33（测试）= $132

### 6.2 存储费用

- S3 存储（DCP、日志）：约 $1/月
- EBS 卷（100GB）：约 $10/月

### 6.3 总成本

**一次完整验证周期**：约 $150-200

---

## 七、风险与对策

### 7.1 时序收敛风险

**风险**：设计在 FPGA 上无法达到 100MHz

**对策**：
- 添加流水线寄存器
- 优化关键路径
- 降低目标频率到 50MHz（仍可验证功能）

### 7.2 资源不足风险

**风险**：设计超出 FPGA 资源

**对策**：
- 使用更大的 F1 实例（f1.16xlarge）
- 简化设计（移除调试逻辑）
- 分模块验证

### 7.3 调试困难风险

**风险**：FPGA 上难以定位问题

**对策**：
- 充分的 RTL 仿真
- 使用 ILA 在线调试
- 添加调试寄存器和状态输出

---

## 八、交付物

### 8.1 文档

- [ ] 环境搭建指南
- [ ] 构建流程文档
- [ ] 测试报告
- [ ] 性能分析报告
- [ ] 问题与解决方案

### 8.2 代码

- [ ] FPGA 顶层封装
- [ ] 约束文件
- [ ] 构建脚本
- [ ] 测试程序
- [ ] 驱动程序

### 8.3 结果

- [ ] 综合报告
- [ ] 时序报告
- [ ] 功耗报告
- [ ] 测试日志
- [ ] 波形文件

---

## 九、时间计划

| 阶段 | 任务 | 工期 |
|-----|------|------|
| 第 1 周 | 环境搭建、设计准备 | 5 天 |
| 第 2 周 | FPGA 综合与实现 | 5 天 |
| 第 3 周 | AFI 创建与部署 | 3 天 |
| 第 3-4 周 | 功能测试 | 5 天 |
| 第 4 周 | 性能测试与优化 | 3 天 |
| 第 5 周 | 文档整理与交付 | 3 天 |

**总工期**：约 5 周

---

## 十、参考资料

### 10.1 AWS 文档

- [AWS F1 实例用户指南](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/fpga-getting-started.html)
- [AWS FPGA HDK](https://github.com/aws/aws-fpga)
- [AFI 创建指南](https://github.com/aws/aws-fpga/blob/master/hdk/docs/create_dcp_from_cl.md)

### 10.2 Xilinx 文档

- [UltraScale+ FPGA 数据手册](https://www.xilinx.com/products/silicon-devices/fpga/virtex-ultrascale-plus.html)
- [Vivado 设计套件用户指南](https://www.xilinx.com/support/documentation-navigation/design-hubs/dh0010-vivado-design-hub.html)

### 10.3 项目文档

- `chisel/README.md` - Chisel 设计说明
- `chisel/synthesis/README.md` - 综合流程说明
- `docs/RISC-V_AI加速器芯片流片说明报告.md` - 芯片设计报告

---

## 附录 A：快速开始

```bash
# 1. 启动 F1 实例
aws ec2 run-instances --instance-type f1.2xlarge ...

# 2. 连接实例
ssh -i key.pem centos@<ip>

# 3. 设置环境
cd aws-fpga && source sdk_setup.sh

# 4. 克隆项目
git clone <your-repo>
cd chisel/synthesis/fpga

# 5. 构建 FPGA
./scripts/setup_aws.sh
./scripts/build_fpga.tcl

# 6. 创建 AFI
./scripts/create_afi.sh

# 7. 加载并测试
sudo fpga-load-local-image -S 0 -I <afi-id>
./scripts/run_tests.sh
```

---

**文档版本**：1.0  
**最后更新**：2025年11月16日  
**维护者**：redoop 团队
