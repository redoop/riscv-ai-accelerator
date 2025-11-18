# 测试启动流程说明

## 问题描述

当运行 `bash run_fpga_flow.sh aws-launch f1` 并选择选项 2（F1 按需实例）时，启动失败。

## 原因分析

`run_fpga_flow.sh` 中的 `aws_launch_instance` 函数参数传递有问题：
- 函数使用 `$2` 获取实例类型参数
- 但在 main 函数中调用时，`$2` 已经是全局参数
- 导致参数混乱

## 修复方案

### 修改 1: 在 main 函数中设置 TARGET

```bash
aws-launch)
    show_banner
    # 设置 TARGET 为第二个参数（f1 或 f2）
    TARGET="${2:-auto}"
    aws_launch_instance
    ;;
```

### 修改 2: 改进 aws_launch_instance 函数

```bash
aws_launch_instance() {
    # 获取实例类型参数（从全局 TARGET）
    local instance_type="${TARGET}"
    if [ "$instance_type" == "aws" ] || [ "$instance_type" == "local" ]; then
        instance_type="auto"
    fi
    
    # ... 处理逻辑
}
```

## 使用方法

### 方式 1: 直接指定 F1（推荐）

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
bash run_fpga_flow.sh aws-launch f1
```

这会自动使用 F1 Spot 实例（最便宜）。

### 方式 2: 交互式选择

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
bash run_fpga_flow.sh aws-launch
```

然后选择：
- 1 = F1 Spot 实例（~$0.50/小时）
- 2 = F1 按需实例（$1.65/小时）
- 3 = F2 实例（不推荐）
- 4 = 查看详细对比

### 方式 3: 直接使用启动脚本

如果 `run_fpga_flow.sh` 仍有问题，可以直接使用启动脚本：

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga/aws-deployment

# F1 Spot 实例（最便宜）
bash launch_f1_vivado.sh

# F1 按需实例（最可靠）
bash launch_f1_ondemand.sh

# 交互式选择
bash launch_fpga_instance.sh
```

## 测试步骤

### 测试 1: F1 Spot 实例

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
bash run_fpga_flow.sh aws-launch f1
```

**预期结果:**
- 显示 "✓ 选择 F1 实例（推荐）"
- 自动尝试多个可用区
- 成功启动 F1 Spot 实例

### 测试 2: 交互式选择 F1 按需

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
bash run_fpga_flow.sh aws-launch
# 输入: 2
# 输入: y (确认)
```

**预期结果:**
- 显示交互式菜单
- 选择 2 后显示按需实例特点
- 确认后启动 F1 按需实例

### 测试 3: 直接使用按需脚本

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga/aws-deployment
bash launch_f1_ondemand.sh
```

**预期结果:**
- 直接启动 F1 按需实例
- 自动尝试多个可用区
- 显示详细的错误信息（如果失败）

## 可用区自动选择

所有脚本都会自动尝试以下可用区（按顺序）：

1. us-east-1a
2. us-east-1b
3. us-east-1d
4. us-east-1e
5. us-east-1c

如果某个可用区失败，会自动尝试下一个，并显示失败原因。

## 常见错误和解决方案

### 错误 1: InsufficientInstanceCapacity

**含义**: 该可用区没有足够的 F1 容量

**解决方案**:
- 脚本会自动尝试下一个可用区
- 如果所有可用区都失败：
  - Spot 实例：改用按需实例
  - 按需实例：稍后重试或联系 AWS 支持

### 错误 2: 所有可用区都失败

**可能原因**:
1. F1 配额为 0（需要申请）
2. 账户权限不足
3. 区域暂时不可用

**检查配额**:
```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-85EED4F7 \
  --region us-east-1
```

**申请配额**:
访问 AWS Service Quotas 控制台申请 F1 实例配额。

### 错误 3: 参数错误

**症状**: 脚本报错 "InvalidParameterValue"

**检查**:
```bash
# 检查密钥
aws ec2 describe-key-pairs --key-names fpga-f2-key --region us-east-1

# 检查安全组
aws ec2 describe-security-groups --group-ids sg-03d27449f82b54360 --region us-east-1
```

## 验证实例启动成功

### 检查实例信息文件

```bash
cat /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

**应该包含**:
- INSTANCE_ID
- PUBLIC_IP
- AVAILABILITY_ZONE
- DEVICE=xcvu9p
- BILLING_TYPE (spot 或 on-demand)

### 测试 SSH 连接

```bash
# 从实例信息文件获取 IP
source /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga/aws-deployment/.f1_instance_info

# 测试连接
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} "echo 'SSH 连接成功'"
```

### 验证 Vivado

```bash
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} "vivado -version"
```

**预期输出**:
```
Vivado v2024.1 (64-bit)
```

## 下一步

实例启动成功后：

```bash
# 1. 准备项目
bash run_fpga_flow.sh prepare

# 2. 上传项目
bash run_fpga_flow.sh aws-upload

# 3. 启动构建
bash run_fpga_flow.sh aws-build

# 4. 监控进度
bash run_fpga_flow.sh aws-monitor

# 5. 下载 DCP
bash run_fpga_flow.sh aws-download-dcp

# 6. 创建 AFI
bash run_fpga_flow.sh aws-create-afi

# 7. 清理实例
bash run_fpga_flow.sh aws-cleanup
```

## 成本提醒

| 操作 | 时间 | F1 Spot | F1 按需 |
|------|------|---------|---------|
| 启动实例 | 1-2 分钟 | $0.02 | $0.05 |
| 构建 | 2-4 小时 | $1.00-2.00 | $3.30-6.60 |
| 测试 | 10-20 分钟 | $0.10 | $0.30 |
| **总计** | **~3 小时** | **~$1.12** | **~$3.65** |

**重要**: 构建完成后立即运行 `bash run_fpga_flow.sh aws-cleanup` 以停止实例！

## 相关文档

- [F1 启动选项说明](./F1_LAUNCH_OPTIONS.md)
- [可用区自动选择](./AVAILABILITY_ZONE_SELECTION.md)
- [F1 vs F2 快速参考](./F1_F2_QUICK_REFERENCE.md)
