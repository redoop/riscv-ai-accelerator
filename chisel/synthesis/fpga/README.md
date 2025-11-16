# RISC-V AI 加速器 FPGA 验证

本目录包含 RISC-V AI 加速器的 FPGA 原型验证方案，支持本地和 AWS F1 云端验证。

## 🚀 快速开始

### 完整流程（从 Chisel 到 FPGA）

```bash
# 步骤 1：生成 Verilog（在 chisel 目录）
cd chisel
./run.sh generate

# 步骤 2：FPGA 验证（在 fpga 目录）
cd synthesis/fpga
./run_fpga_flow.sh status      # 查看状态
./run_fpga_flow.sh full local  # 本地验证

# 步骤 3：查看结果
ls -lh ../../../chisel/generated/simple_edgeaisoc/
```

### 如果 Verilog 已生成（快速验证）

```bash
cd chisel/synthesis/fpga

# 1. 查看当前状态
./run_fpga_flow.sh status

# 2. 本地完整流程（仿真 + 综合）
./run_fpga_flow.sh full local

# 3. 查看结果
ls -lh ../../../chisel/generated/simple_edgeaisoc/
```

### 常用命令

```bash
./run_fpga_flow.sh help       # 查看帮助
./run_fpga_flow.sh status     # 查看状态
./run_fpga_flow.sh prepare    # 生成 Verilog
./run_fpga_flow.sh simulate   # 运行仿真
./run_fpga_flow.sh synthesize # 综合设计
./run_fpga_flow.sh clean      # 清理文件
./run_fpga_flow.sh aws        # AWS F1 完整流程
```

## 📁 目录结构

```
fpga/
├── run_fpga_flow.sh          # 统一的自动化脚本 ⭐
├── README.md                  # 本文件（主文档）
├── AWS_FPGA_PLAN.md          # 完整的 AWS FPGA 验证方案
├── constraints/               # FPGA 约束文件
│   ├── timing.xdc            # 时序约束（100 MHz）
│   ├── pins.xdc              # 引脚约束（AWS Shell）
│   └── physical.xdc          # 物理约束（布局优化）
├── scripts/                   # 自动化脚本
│   ├── setup_aws.sh          # AWS 环境配置
│   ├── build_fpga.tcl        # Vivado 构建脚本
│   ├── create_afi.sh         # AFI 创建脚本
│   └── run_tests.sh          # 测试脚本
├── src/                       # FPGA 适配层源码
│   ├── fpga_top.v            # FPGA 顶层封装
│   ├── clock_gen.v           # 时钟生成（MMCM）
│   └── io_adapter.v          # IO 适配器（PCIe BAR）
├── docs/                      # 详细文档
│   └── SETUP_GUIDE.md        # AWS 环境搭建指南
├── build/                     # 构建输出（自动生成）
└── test_results/              # 测试结果（自动生成）
```

## 🎯 验证流程

### 方案 1：本地验证（推荐，免费）✅

**适用场景：** 日常开发、快速迭代、功能验证

```bash
./run_fpga_flow.sh full local
```

**执行步骤：**
1. 检查依赖（sbt, java）
2. 生成 Verilog（3765 行）
3. 运行 RTL 仿真测试
4. Yosys 综合（可选）

**时间：** 10-20 分钟 | **成本：** 免费

**验证结果：**
```bash
$ ./run_fpga_flow.sh full local

[1/7] 检查依赖...
✓ 所有依赖已安装

[2/7] 准备环境...
✓ 目录结构创建完成

[3/7] 生成 Verilog...
✓ Verilog 生成成功 (3765 行)

[4/7] 运行 RTL 仿真...
✓ 仿真测试通过

╔════════════════════════════════════════════════════════════╗
║  本地流程完成！                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 方案 2：AWS F1 验证（完整硬件验证）⏳

**适用场景：** 最终验证、性能测试、硬件部署

**⚠️ 注意：** 此流程需要在 AWS F1 实例上运行，本地环境会提示缺少 AWS CLI。

**完整步骤：**

#### 步骤 1：本地准备（本地机器）

```bash
# 生成 Verilog
./run_fpga_flow.sh prepare

# 打包项目
cd ../..
tar czf fpga-project.tar.gz synthesis/fpga/ generated/
```

#### 步骤 2：启动 F1 实例（本地机器）

```bash
# 安装 AWS CLI（如果未安装）
pip install awscli

# 配置 AWS 凭证
aws configure

# 启动 f1.2xlarge 实例
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type f1.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx

# 获取实例 IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=FPGA-Dev" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "实例 IP: $INSTANCE_IP"
```

#### 步骤 3：上传项目（本地机器）

```bash
# 上传项目
scp -i ~/.ssh/your-key.pem fpga-project.tar.gz centos@$INSTANCE_IP:~/

# 连接到实例
ssh -i ~/.ssh/your-key.pem centos@$INSTANCE_IP
```

#### 步骤 4：在 F1 实例上运行（F1 实例）

```bash
# 解压项目
tar xzf fpga-project.tar.gz
cd synthesis/fpga

# 配置 AWS 环境
./scripts/setup_aws.sh
source ~/.fpga_config

# 运行完整 AWS 流程
./run_fpga_flow.sh aws
```

**执行步骤：**
1. Vivado 综合和实现（2-4 小时）
2. 创建 AFI 镜像（30-60 分钟）
3. 等待 AFI 可用
4. 部署到 FPGA
5. 运行硬件测试

**时间：** 3-5 小时 | **成本：** $150-200

**预期输出：**
```bash
╔════════════════════════════════════════════════════════════╗
║  AWS F1 流程完成！                                         ║
╚════════════════════════════════════════════════════════════╝

下一步:
  1. 等待 AFI 生成完成 (30-60 分钟)
  2. 检查状态: ./run_fpga_flow.sh status
  3. 部署测试: ./run_fpga_flow.sh deploy aws
  4. 运行测试: ./run_fpga_flow.sh test
```

### 验证检查点

| 阶段 | 检查命令 | 成功标志 |
|------|---------|---------|
| Verilog 生成 | `./run_fpga_flow.sh status` | ✓ Verilog 已生成 (3765 行) |
| RTL 仿真 | `./run_fpga_flow.sh simulate` | ✓ 仿真测试通过 |
| Vivado 综合 | `ls build/checkpoints/` | 存在 SH_CL_routed.dcp |
| AFI 创建 | `cat build/afi_info.txt` | 显示 AFI ID 和 AGFI ID |
| FPGA 测试 | `./run_fpga_flow.sh test` | 所有测试通过 |

## 📊 设计信息

**SoC 组成：**
- PicoRV32 RISC-V 处理器（RV32I）
- CompactAccel 8x8 矩阵加速器
- BitNetAccel 16x16 BitNet 加速器
- UART、GPIO 外设
- 中断控制器

**性能指标：**
- 工作频率：100 MHz
- 峰值性能：6.4 GOPS
- 功耗：< 100 mW（ASIC）

**资源估算（FPGA）：**
- LUT：~50,000（VU9P 的 4%）
- FF：~40,000（VU9P 的 2%）
- BRAM：~20（VU9P 的 1%）

## 🔧 前置条件和平台支持

### 支持的平台

| 平台 | 本地验证 | AWS F1 验证 | 说明 |
|------|---------|------------|------|
| **macOS** | ✅ 完全支持 | ✅ 支持 | 推荐用于日常开发 |
| **Linux** | ✅ 完全支持 | ✅ 支持 | 服务器环境 |
| **Windows** | ⚠️ WSL2 | ✅ 支持 | 需要 WSL2 或虚拟机 |

### 本地开发环境

**必需组件：**
- Java 11+
- sbt（Scala 构建工具）

**可选组件：**
- Yosys（用于本地综合，可选）
- AWS CLI（用于 AWS F1 部署）

**安装命令：**

```bash
# macOS（推荐）
brew install openjdk@11 sbt
brew install awscli  # 可选
brew install yosys   # 可选

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install openjdk-11-jdk sbt
pip3 install awscli  # 可选
sudo apt install yosys  # 可选

# Windows (WSL2)
# 先安装 WSL2，然后按 Linux 步骤操作
```

**当前测试环境：**
- ✅ macOS Sonoma (Apple Silicon)
- ✅ Java 11+
- ✅ sbt 1.x
- ✅ AWS CLI 2.31.37
- ⚠️ Yosys 0.56（不支持 SystemVerilog automatic）

### AWS F1 验证环境

**必需：**
- AWS 账户（支持 F1 实例）
- AWS CLI 已配置
- Vivado 2021.2+（F1 实例预装）

**推荐实例：**
- f1.2xlarge（开发测试）
- f1.4xlarge（性能测试）

## 📊 完整开发流程

### 流程图

```
1. Chisel 设计 (chisel/src/)
    ↓
2. 生成 Verilog (chisel/run.sh generate)
    ↓
3. Verilog 输出 (chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv)
    ↓
4. FPGA 验证 (chisel/synthesis/fpga/run_fpga_flow.sh)
    ├─ 本地仿真 (Verilator)
    └─ AWS F1 综合 (Vivado)
```

### 当前状态

运行 `./run_fpga_flow.sh status` 查看：

```bash
$ ./run_fpga_flow.sh status

╔════════════════════════════════════════════════════════════╗
║   RISC-V AI 加速器 - FPGA 验证流程                        ║
╚════════════════════════════════════════════════════════════╝

项目状态:

✓ Verilog 已生成 (3765 行)
○ Vivado 综合未完成
○ AFI 未创建
○ 无测试结果

文件位置:
  Verilog:  ../../../chisel/generated/simple_edgeaisoc/
  构建:     build/
  脚本:     scripts/
  文档:     docs/
```

### Verilog 生成说明

**生成命令：**
```bash
cd ../../../chisel
./run.sh generate
```

**生成的文件：**
- `chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv` (3765 行)
- 包含完整的 SoC 设计：PicoRV32 + CompactAccel + BitNetAccel + PicoRV32 核心代码

**注意：** 
- FPGA 验证脚本会自动使用已生成的 Verilog 文件
- 如果修改了 Chisel 设计，需要重新运行 `chisel/run.sh generate`
- Verilog 文件包含 SystemVerilog 特性（如 `automatic` 关键字）

### 已完成 ✅

| 项目 | 状态 | 平台 | 说明 |
|------|------|------|------|
| Chisel RTL 设计 | ✅ | All | PicoRV32 + CompactAccel + BitNetAccel |
| Verilog 生成 | ✅ | All | SimpleEdgeAiSoC.sv (3765 行) |
| RTL 仿真验证 | ✅ | All | ChiselTest 测试通过 |
| FPGA 适配层 | ✅ | All | fpga_top.v, clock_gen.v, io_adapter.v |
| 约束文件 | ✅ | All | timing.xdc, pins.xdc, physical.xdc |
| 自动化脚本 | ✅ | All | run_fpga_flow.sh (7 种模式) |
| 文档 | ✅ | All | README + AWS_FPGA_PLAN + SETUP_GUIDE |
| macOS 测试 | ✅ | macOS | 本地验证完全通过 |
| AWS CLI 安装 | ✅ | macOS | 版本 2.31.37 |
| 本地流程测试 | ✅ | macOS | 前 4 步验证通过 |

### 待完成（需要 AWS F1 实例）⏳

| 项目 | 状态 | 预计时间 | 成本 |
|------|------|---------|------|
| Vivado 综合 | ⏳ | 2-4 小时 | $3-7 |
| AFI 创建 | ⏳ | 30-60 分钟 | $1-2 |
| FPGA 部署 | ⏳ | 5 分钟 | $0.1 |
| 硬件测试 | ⏳ | 10-20 分钟 | $0.3-0.6 |
| **总计** | - | **3-5 小时** | **$150-200** |

### 验证路径

```
本地开发环境 ✅
    ↓
[已完成] Verilog 生成 ✅
    ↓
[已完成] RTL 仿真 ✅
    ↓
[待完成] AWS F1 实例 ⏳
    ↓
[待完成] Vivado 综合 ⏳
    ↓
[待完成] AFI 创建 ⏳
    ↓
[待完成] FPGA 部署 ⏳
    ↓
[待完成] 硬件测试 ⏳
```

## 📚 文档

- **README.md**（本文件）- 快速开始和完整指南
- **AWS_FPGA_PLAN.md** - 详细的 AWS FPGA 验证方案（技术细节、成本分析、时间计划）
- **docs/SETUP_GUIDE.md** - AWS 环境搭建步骤（账户配置、实例启动、工具安装）
- **../../../docs/RISC-V_AI加速器芯片流片说明报告.md** - 芯片设计完整报告

## 🐛 常见问题

### Q1: 本地运行 `./run_fpga_flow.sh aws` 提示缺少 aws-cli

**问题：**
```bash
$ ./run_fpga_flow.sh aws
❌ 缺少依赖: aws-cli
```

**解决方案：**
```bash
# macOS
brew install awscli

# Linux
pip3 install awscli --user

# 验证安装
aws --version
```

**测试结果（本地环境）：**
```bash
$ ./run_fpga_flow.sh aws

[1/7] 检查依赖...
✓ 所有依赖已安装

[2/7] 准备环境...
✓ 目录结构创建完成

[3/7] 生成 Verilog...
✓ Verilog 生成成功 (3765 行)

[4/7] 运行 RTL 仿真...
✓ 仿真测试通过

[5/7] AWS Vivado 综合...
❌ Vivado 未安装
  请在 AWS F1 实例上运行此脚本
```

**说明：** 本地环境可以完成前 4 步（依赖检查、环境准备、Verilog 生成、RTL 仿真），第 5 步开始需要在 AWS F1 实例上运行。

### Q2: Verilog 生成失败

**问题：**
```bash
❌ Verilog 生成失败
```

**解决方案：**
```bash
# 检查依赖
java -version  # 需要 Java 11+
sbt --version  # 需要 sbt

# 重新生成
cd ../../../chisel
./run.sh generate
```

### Q3: 仿真测试失败

**问题：**
```bash
✗ 仿真测试失败
```

**解决方案：**
```bash
# 查看详细日志
cd ../../../chisel
./run.sh soc

# 检查测试目录
ls -la test_run_dir/SimpleEdgeAiSoC*

# 查看波形文件
ls -la test_run_dir/SimpleEdgeAiSoC*/SimpleEdgeAiSoC.vcd
```

### Q4: AWS 权限不足

**问题：**
```bash
UnauthorizedOperation: You are not authorized to perform this operation
```

**解决方案：**
```bash
# 检查 AWS 配置
aws configure list
aws sts get-caller-identity

# 检查 F1 实例限额
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-85EED4F7

# 如果限额为 0，需要申请增加
# 在 AWS 控制台 -> Service Quotas -> EC2 -> 搜索 "F1"
```

### Q5: 时序不收敛（WNS < 0）

**问题：**
```bash
WNS (Worst Negative Slack): -2.345 ns
```

**解决方案：**

编辑 `constraints/timing.xdc`，降低目标频率：
```tcl
# 从 100 MHz (10.000ns) 改为 50 MHz (20.000ns)
create_clock -period 20.000 -name sys_clk [get_ports clock]
```

或者优化关键路径：
```bash
# 查看时序报告
cat build/reports/timing_impl.rpt

# 找到关键路径并优化
```

### Q6: AFI 创建失败

**问题：**
```bash
AFI 状态: failed
```

**解决方案：**
```bash
# 查看 AFI 日志
AFI_ID=$(grep "AFI ID" build/afi_info.txt | awk '{print $3}')
S3_BUCKET=$(grep "S3 Bucket" build/afi_info.txt | awk '{print $3}')

# 下载日志
aws s3 ls s3://$S3_BUCKET/logs/
aws s3 cp s3://$S3_BUCKET/logs/ ./logs/ --recursive

# 查看错误信息
cat logs/*.log
```

### Q7: 如何查看构建日志

```bash
# Vivado 日志
cat build/vivado.log

# 测试日志
ls test_results/

# AFI 信息
cat build/afi_info.txt

# 实时查看 Vivado 进度
tail -f build/vivado.log
```

### Q8: 本地 Yosys 综合失败（syntax error, unexpected TOK_AUTOMATIC）

**问题：**
```bash
ERROR: syntax error, unexpected TOK_AUTOMATIC
```

**原因：** Chisel 生成的 SystemVerilog 使用了 `automatic` 关键字，Yosys 的标准 Verilog 前端不支持。

**解决方案：**

**方案 1：跳过本地综合（推荐）**
```bash
# 只做 RTL 验证，跳过综合
./run_fpga_flow.sh prepare
./run_fpga_flow.sh simulate
```

本地综合是可选的，主要用于快速检查。真正的 FPGA 综合应该在 AWS F1 上使用 Vivado 完成。

**方案 2：使用 Yosys slang 插件**
```bash
# 安装 slang 插件（需要从源码编译）
# 参考：https://github.com/YosysHQ/yosys-slang

# 或者使用 oss-cad-suite（包含 slang）
# https://github.com/YosysHQ/oss-cad-suite-build
```

**方案 3：直接使用 AWS F1**
```bash
# 在 AWS F1 实例上使用 Vivado 综合
# Vivado 完全支持 SystemVerilog
./run_fpga_flow.sh aws
```

**说明：** 本地 Yosys 综合失败不影响整体流程，RTL 仿真已经验证了功能正确性。

## 💡 使用场景和最佳实践

### 使用场景选择

| 场景 | 推荐方案 | 命令 | 时间 | 成本 |
|------|---------|------|------|------|
| 🔧 日常开发 | 本地验证 | `./run_fpga_flow.sh full local` | 10-20 分钟 | 免费 |
| 🧪 功能测试 | 本地验证 | `./run_fpga_flow.sh simulate` | 2-5 分钟 | 免费 |
| 📝 代码修改后验证 | 本地验证 | `./run_fpga_flow.sh prepare && ./run_fpga_flow.sh simulate` | 5 分钟 | 免费 |
| 🎯 最终验证 | AWS F1 | `./run_fpga_flow.sh aws` | 3-5 小时 | $150-200 |
| 📊 性能测试 | AWS F1 | 在 F1 上运行测试 | 10-20 分钟 | $0.3-0.6 |
| 🎓 学习实验 | 本地验证 | `./run_fpga_flow.sh full local` | 10-20 分钟 | 免费 |
| 🚀 产品演示 | AWS F1 | 提前准备好 AFI | 5 分钟 | $0.1 |

### 开发流程建议

```
第 1 阶段：本地开发（1-2 周）
├─ 设计修改
├─ 本地验证（./run_fpga_flow.sh full local）
├─ 功能测试
└─ 迭代优化

第 2 阶段：AWS 验证（1 天）
├─ 启动 F1 实例
├─ 运行完整流程（./run_fpga_flow.sh aws）
├─ 等待 AFI 生成
└─ 硬件测试

第 3 阶段：性能优化（可选）
├─ 分析测试结果
├─ 本地修改优化
└─ 再次 AWS 验证
```

### 成本优化建议

1. **本地优先**：尽量在本地完成所有功能验证
2. **批量验证**：积累多个修改后，一次性在 AWS 上验证
3. **使用 Spot 实例**：可节省 70% 成本
4. **及时停止实例**：验证完成后立即停止实例
5. **复用 AFI**：同一设计的 AFI 可以重复使用

### 时间管理建议

1. **并行工作**：Vivado 综合时（2-4 小时），可以做其他工作
2. **提前规划**：AFI 创建需要 30-60 分钟，提前启动
3. **分阶段验证**：不要等到最后才做 AWS 验证
4. **自动化**：使用脚本自动化重复性工作

### 版本控制建议

```bash
# 每次 AWS 验证前打 tag
git tag -a v1.0-fpga-verify -m "FPGA verification v1.0"
git push origin v1.0-fpga-verify

# 记录 AFI ID
echo "v1.0: afi-xxxxxxxxx" >> afi_versions.txt
```

## 📞 获取帮助

**文档：**
- 快速问题：本 README
- 详细方案：AWS_FPGA_PLAN.md
- 环境搭建：docs/SETUP_GUIDE.md

**命令：**
```bash
./run_fpga_flow.sh help    # 查看帮助
./run_fpga_flow.sh status  # 查看状态
```

**外部资源：**
- AWS F1：https://docs.aws.amazon.com/ec2/latest/userguide/fpga.html
- AWS FPGA GitHub：https://github.com/aws/aws-fpga
- Chisel：https://www.chisel-lang.org/

## 📋 快速参考

### 命令速查表

| 命令 | 功能 | 时间 | 环境 |
|------|------|------|------|
| `./run_fpga_flow.sh help` | 显示帮助 | 1s | 本地 |
| `./run_fpga_flow.sh status` | 查看状态 | 1s | 本地 |
| `./run_fpga_flow.sh prepare` | 生成 Verilog | 2-5 分钟 | 本地 |
| `./run_fpga_flow.sh simulate` | 运行仿真 | 2-5 分钟 | 本地 |
| `./run_fpga_flow.sh synthesize local` | 本地综合 | 5-10 分钟 | 本地 |
| `./run_fpga_flow.sh full local` | 本地完整流程 | 10-20 分钟 | 本地 |
| `./run_fpga_flow.sh clean` | 清理文件 | 1s | 本地 |
| `./run_fpga_flow.sh aws` | AWS 完整流程 | 3-5 小时 | F1 实例 |
| `./run_fpga_flow.sh deploy aws` | 部署 AFI | 1 分钟 | F1 实例 |
| `./run_fpga_flow.sh test` | 运行测试 | 10-20 分钟 | F1 实例 |

### 文件位置速查

| 文件类型 | 位置 |
|---------|------|
| 生成的 Verilog | `../../../chisel/generated/simple_edgeaisoc/` |
| 构建输出 | `build/` |
| 测试结果 | `test_results/` |
| Vivado 日志 | `build/vivado.log` |
| AFI 信息 | `build/afi_info.txt` |
| 时序报告 | `build/reports/timing_impl.rpt` |
| 资源报告 | `build/reports/utilization_impl.rpt` |

### 关键指标速查

| 指标 | 目标值 | 检查方法 |
|------|--------|---------|
| Verilog 行数 | 3765 | `wc -l ../../../chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv` |
| 工作频率 | 100 MHz | `grep "WNS" build/reports/timing_impl.rpt` |
| LUT 使用 | ~50,000 (4%) | `grep "Slice LUTs" build/reports/utilization_impl.rpt` |
| BRAM 使用 | ~20 (1%) | `grep "Block RAM" build/reports/utilization_impl.rpt` |
| 峰值性能 | 6.4 GOPS | 运行性能测试 |

### 故障排查速查

| 问题 | 快速检查 | 解决方案 |
|------|---------|---------|
| Verilog 未生成 | `ls ../../../chisel/generated/` | `./run_fpga_flow.sh prepare` |
| 仿真失败 | `ls ../../../chisel/test_run_dir/` | 查看测试日志 |
| AWS CLI 缺失 | `aws --version` | `pip install awscli` |
| 时序不收敛 | `grep WNS build/reports/timing_impl.rpt` | 降低频率或优化 |
| AFI 创建失败 | `cat build/afi_info.txt` | 查看 S3 日志 |

---

**版本**：1.0  
**更新时间**：2025年11月16日  
**维护者**：redoop 团队  
**文档状态**：✅ 已完成并测试
