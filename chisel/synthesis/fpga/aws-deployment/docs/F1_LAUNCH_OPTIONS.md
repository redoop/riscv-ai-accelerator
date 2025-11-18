# F1 实例启动选项说明

## 概述

本项目提供三种方式启动 F1 实例，满足不同场景需求。

## 启动脚本对比

| 脚本 | 实例类型 | 成本 | 可用性 | 适用场景 |
|------|---------|------|--------|---------|
| `launch_f1_vivado.sh` | F1 Spot | ~$0.50/小时 | 可能不足 | 成本敏感项目 |
| `launch_f1_ondemand.sh` | F1 按需 | $1.65/小时 | 保证可用 | 可靠性优先 |
| `launch_fpga_instance.sh` | 交互选择 | 取决于选择 | 取决于选择 | 不确定选哪个 |

## 使用方法

### 方式 1: 直接启动 F1 Spot 实例（最便宜）

```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_f1_vivado.sh
```

**优点:**
- 成本最低（~$0.50/小时）
- 4小时构建仅需 $2.00

**缺点:**
- 可能因容量不足而失败
- 可能被中断（极少见）

**适合:**
- 成本敏感的项目
- 可以接受偶尔失败重试
- 非紧急任务

### 方式 2: 直接启动 F1 按需实例（最可靠）

```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_f1_ondemand.sh
```

**优点:**
- 保证可用，不会因容量不足而失败
- 不会被中断
- 适合长时间构建

**缺点:**
- 成本是 Spot 的 3 倍（$1.65/小时）
- 4小时构建需要 $6.60

**适合:**
- 需要保证成功的任务
- 紧急项目
- 生产环境
- 长时间构建（避免中断）

### 方式 3: 交互式选择（推荐新手）

```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_fpga_instance.sh
```

**功能:**
- 提供交互式菜单
- 显示详细对比信息
- 支持选择 F1 Spot、F1 按需或 F2
- 包含确认步骤

**适合:**
- 第一次使用
- 不确定选哪个
- 需要查看详细对比

## 决策树

```
需要保证成功？
├─ 是 → 使用 launch_f1_ondemand.sh（按需实例）
└─ 否 → 
    ├─ 成本敏感？
    │   ├─ 是 → 使用 launch_f1_vivado.sh（Spot 实例）
    │   └─ 否 → 使用 launch_f1_ondemand.sh（按需实例）
    └─ 不确定？→ 使用 launch_fpga_instance.sh（交互选择）
```

## 成本对比

### 4小时构建任务

| 实例类型 | 小时费率 | 4小时成本 | 节省 |
|---------|---------|----------|------|
| F1 Spot | $0.50 | $2.00 | 基准 |
| F1 按需 | $1.65 | $6.60 | -$4.60 |

### 24小时开发任务

| 实例类型 | 小时费率 | 24小时成本 | 节省 |
|---------|---------|-----------|------|
| F1 Spot | $0.50 | $12.00 | 基准 |
| F1 按需 | $1.65 | $39.60 | -$27.60 |

## 实例信息

所有脚本都使用相同的配置：

- **AMI**: ami-092fc5deb8f3c0f7d
- **AMI 名称**: FPGA Developer AMI (Ubuntu) 1.16.1
- **Vivado 版本**: 2024.1
- **实例类型**: f1.2xlarge
- **FPGA 设备**: xcvu9p (与 AWS AFI 服务兼容)
- **区域**: us-east-1
- **密钥**: fpga-f2-key

## 常见问题

### Q: Spot 实例会被中断吗？

A: F1 Spot 实例被中断的概率很低（<5%），但在容量紧张时可能发生。如果担心中断，建议使用按需实例。

### Q: Spot 实例启动失败怎么办？

A: 脚本会自动尝试多个可用区。如果所有可用区都失败，可以：
1. 提高 Spot 出价
2. 稍后重试
3. 使用按需实例（`launch_f1_ondemand.sh`）

### Q: 如何选择 Spot 还是按需？

A: 简单规则：
- **成本优先** → Spot 实例
- **可靠性优先** → 按需实例
- **不确定** → 先试 Spot，失败再用按需

### Q: 两个脚本生成的 DCP 有区别吗？

A: 没有区别。两者使用相同的 AMI、Vivado 版本和 FPGA 设备（xcvu9p），生成的 DCP 完全相同，都可以用于创建 AFI。

### Q: 实例信息保存在哪里？

A: 所有脚本都将实例信息保存到 `.f1_instance_info` 文件，包含：
- 实例 ID
- 公网 IP
- 可用区
- 计费类型（Spot 或 On-Demand）
- 时间戳

## 后续步骤

启动实例后，按照以下步骤继续：

```bash
# 1. 上传项目
./run_fpga_flow.sh aws-upload

# 2. 启动构建
./run_fpga_flow.sh aws-build

# 3. 监控进度
./run_fpga_flow.sh aws-monitor

# 4. 下载 DCP
./run_fpga_flow.sh aws-download-dcp

# 5. 创建 AFI
./run_fpga_flow.sh aws-create-afi

# 6. 清理实例（重要！）
./run_fpga_flow.sh aws-cleanup
```

## 注意事项

⚠️ **重要提醒:**

1. **及时清理**: 构建完成后立即停止实例，避免持续计费
2. **监控成本**: 定期检查 AWS 账单
3. **备份数据**: 停止实例前确保已下载所有需要的文件
4. **F1 兼容**: 只有 F1 实例（xcvu9p）生成的 DCP 可以创建 AFI

## 相关文档

- [F1 vs F2 快速参考](./F1_F2_QUICK_REFERENCE.md)
- [F1 README](./f1/README.md)
- [AWS FPGA HDK](../aws-fpga/aws-fpga/hdk/README.md)
