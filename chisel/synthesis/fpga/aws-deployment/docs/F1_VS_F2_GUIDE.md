# F1 vs F2 实例选择指南

## 快速决策

**需要创建 AFI？** → 使用 **F1**  
**仅本地开发？** → 可以使用 F2

## 详细对比

### F1 实例（推荐用于 AFI）

| 特性 | 详情 |
|------|------|
| **FPGA 设备** | xcvu9p (Virtex UltraScale+ VU9P) |
| **逻辑单元** | 2,586,000 LUTs |
| **Block RAM** | 6,840 个 (38.5 Mb) |
| **DSP** | 6,840 个 |
| **AFI 兼容性** | ✅ 完全支持 |
| **实例类型** | f1.2xlarge, f1.4xlarge, f1.16xlarge |
| **Spot 价格** | ~$0.50-0.60/小时 (f1.2xlarge) |
| **按需价格** | $1.65/小时 (f1.2xlarge) |
| **推荐用途** | AFI 创建、测试、生产部署 |

### F2 实例（仅用于开发）

| 特性 | 详情 |
|------|------|
| **FPGA 设备** | xcvu47p (Virtex UltraScale+ VU47P) |
| **逻辑单元** | 9,024,000 LUTs |
| **Block RAM** | 11,520 个 (82.1 Mb) |
| **DSP** | 4,272 个 |
| **AFI 兼容性** | ❌ 不支持 |
| **实例类型** | f2.2xlarge, f2.6xlarge, f2.48xlarge |
| **Spot 价格** | ~$2.30/小时 (f2.6xlarge) |
| **按需价格** | $7.65/小时 (f2.6xlarge) |
| **推荐用途** | 本地开发、调试（不创建 AFI） |

## 为什么 F2 不支持 AFI？

### 设备不匹配问题

```
F1 DCP:  xcvu9p  ✓ → AWS Shell (xcvu9p) → AFI ✓
F2 DCP:  xcvu47p ✗ → AWS Shell (xcvu9p) → 错误 ✗
```

AWS AFI 服务的 Shell 是为 xcvu9p 设计的，无法接受 xcvu47p 的 DCP。

### 错误示例

```
ERROR: [Constraints 18-884] HDPRVerify-01: 
design check point is using device xcvu47p, 
yet AWS Shell is using device xcvu9p.
```

## 使用场景

### 场景 1：创建 AFI 并部署到 F1

**目标**：创建可在 F1 实例上运行的 AFI

**流程**：
```bash
# 1. 使用 F1 构建
./run_fpga_flow.sh aws-launch  # 选择 F1
./run_fpga_flow.sh aws-build

# 2. 创建 AFI
./run_fpga_flow.sh aws-create-afi

# 3. 在 F1 上测试
# 启动 F1 测试实例
# 加载 AFI
# 运行测试
```

**成本**：
- F1 构建 (4小时): $2.00 (Spot)
- AFI 创建: 免费
- F1 测试 (30分钟): $0.25 (Spot)
- **总计**: ~$2.25

### 场景 2：本地开发（不需要 AFI）

**目标**：开发和调试设计，不部署到云端

**流程**：
```bash
# 使用 F2 进行开发
./run_fpga_flow.sh aws-launch  # 选择 F2
./run_fpga_flow.sh aws-build

# 下载 DCP 用于本地分析
./run_fpga_flow.sh aws-download-dcp

# 不创建 AFI
```

**成本**：
- F2 构建 (4小时): $9.20 (Spot)
- **总计**: ~$9.20

**注意**：F2 更贵且不能创建 AFI，不推荐

### 场景 3：混合使用（不推荐）

**问题**：在 F2 上构建，然后尝试创建 AFI

**结果**：❌ 失败

```bash
# 在 F2 上构建
./run_fpga_flow.sh aws-launch  # F2
./run_fpga_flow.sh aws-build   # 生成 xcvu47p DCP

# 尝试创建 AFI
./run_fpga_flow.sh aws-create-afi  # ❌ 失败！
# 错误: 设备不匹配 (xcvu47p vs xcvu9p)
```

## 推荐配置

### 开发阶段

**选项 A：全部使用 F1（推荐）**
```bash
# 开发、构建、测试都在 F1
实例: f1.2xlarge Spot
成本: ~$0.50/小时
优点: 一致性好，可直接创建 AFI
```

**选项 B：本地开发 + F1 构建**
```bash
# 本地开发（Chisel/Verilog）
# F1 构建和测试
成本: 仅构建时付费
优点: 成本最低
```

### 生产部署

```bash
# 1. F1 构建 DCP
实例: f1.2xlarge Spot
时间: 2-4 小时
成本: $1-2

# 2. 创建 AFI
服务: AWS AFI
时间: 30-60 分钟
成本: 免费

# 3. F1 测试
实例: f1.2xlarge 按需
时间: 按需
成本: $1.65/小时
```

## 启动脚本

### 使用智能选择脚本（推荐）

```bash
cd aws-deployment
bash launch_fpga_instance.sh
# 会提示选择 F1 或 F2
```

### 直接启动 F1

```bash
cd aws-deployment
bash launch_f1_vivado.sh
```

### 直接启动 F2

```bash
cd aws-deployment
bash launch_f2_vivado.sh
```

## 常见问题

### Q1: 我应该选择哪个？

**A**: 如果需要创建 AFI，必须选择 F1。F2 仅用于不需要 AFI 的本地开发。

### Q2: F2 的 DCP 能转换为 F1 兼容吗？

**A**: 不能。设备不同，必须在 F1 上重新构建。

### Q3: F2 有什么优势？

**A**: F2 有更多资源（9M vs 2.5M LUTs），但：
- 不支持 AFI
- 更贵
- 对于大多数设计，F1 的资源已足够

### Q4: 成本对比

**4小时构建 + 30分钟测试**：

| 配置 | 成本 |
|------|------|
| F1 Spot | $2.25 |
| F1 按需 | $7.45 |
| F2 Spot | $9.45 (无法创建 AFI) |
| F2 按需 | $31.43 (无法创建 AFI) |

**推荐**: F1 Spot

### Q5: 如何验证实例类型？

```bash
# 在实例上运行
ssh -i ~/.ssh/your-key.pem ec2-user@<IP>

# F1 实例
lspci | grep Xilinx
# 应该显示: xcvu9p

# F2 实例
lspci | grep Xilinx
# 应该显示: xcvu47p
```

## 总结

| 需求 | 推荐实例 | 原因 |
|------|----------|------|
| 创建 AFI | F1 | 唯一选择 |
| 测试 AFI | F1 | 唯一选择 |
| 本地开发 | F1 | 更便宜，可选创建 AFI |
| 大型设计 | F1 | F1 资源通常足够 |
| 预算有限 | F1 Spot | 最便宜 |

**结论**：除非有特殊原因，始终使用 F1。
