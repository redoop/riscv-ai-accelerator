# F1 实例容量不足问题

## 当前状况

**问题**: us-east-1 区域的 F1 实例容量严重不足

### 测试结果

| 可用区 | Spot 实例 | 按需实例 | 状态 |
|--------|-----------|----------|------|
| us-east-1a | ❌ 无容量 | ❓ 未测试 | 容量不足 |
| us-east-1b | ❌ 无容量 | ❓ 未测试 | 容量不足 |
| us-east-1c | ⏳ 超时 | ❌ 不支持 | 不支持 F1 |
| us-east-1d | ❓ 未测试 | ❓ 未测试 | 未知 |
| us-east-1e | ❓ 未测试 | ❓ 未测试 | 未知 |

## 根本原因

F1 实例是专用的 FPGA 实例，数量有限：
1. **硬件限制**: F1 使用特定的 FPGA 硬件（Xilinx VU9P）
2. **需求高**: FPGA 开发需求大，容量经常不足
3. **区域限制**: 不是所有区域都有 F1 实例

## 解决方案

### 方案 1：使用其他区域（推荐）

F1 实例在以下区域可能有更好的可用性：

#### us-west-2 (Oregon) - 推荐
```bash
# 更新脚本使用 us-west-2
REGION="us-west-2"
AVAILABILITY_ZONES=("us-west-2a" "us-west-2b" "us-west-2c")
```

**优点**:
- 通常有更好的 F1 容量
- 价格相同
- 延迟略高但可接受

#### eu-west-1 (Ireland)
```bash
REGION="eu-west-1"
AVAILABILITY_ZONES=("eu-west-1a" "eu-west-1b" "eu-west-1c")
```

**注意**: 需要查找该区域的 AMI ID

### 方案 2：使用 F2 实例进行开发

虽然 F2 不支持 AFI 创建，但可以用于：
- 设计开发和验证
- Vivado 综合和实现
- 生成 DCP 文件

**流程**:
1. 在 F2 上开发和构建（xcvu47p）
2. 使用 AWS HDK 官方流程创建 AFI
3. 在 F1 上测试 AFI

```bash
# 使用 F2 进行开发
./run_fpga_flow.sh aws-launch f2
```

**限制**: 
- F2 生成的 DCP 无法直接用于 AFI
- 需要使用 AWS HDK 示例作为模板

### 方案 3：等待容量恢复

F1 容量会动态变化：
- 高峰时段（工作日白天）容量更紧张
- 非高峰时段（晚上、周末）可能有容量
- 定期检查可用性

```bash
# 运行可用性检查
bash check_f1_availability.sh

# 每小时检查一次
watch -n 3600 'bash check_f1_availability.sh'
```

### 方案 4：请求配额增加

如果你有长期需求，可以请求专用容量：

1. **访问 AWS Service Quotas**
   https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas

2. **请求增加 F1 配额**
   - 配额代码: L-74FC7D96
   - 说明你的使用场景
   - 通常需要 1-3 个工作日

3. **考虑预留实例**
   - 如果需要长期使用
   - 可以保证容量
   - 有折扣

### 方案 5：使用 AWS FPGA HDK 官方流程

不使用自己的 DCP，而是使用 AWS 提供的示例：

```bash
# 1. 等待 F1 容量
# 2. 启动 F1 实例
# 3. 使用 AWS HDK 示例
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source hdk_setup.sh

# 4. 使用 cl_hello_world 示例
cd hdk/cl/examples/cl_hello_world
export CL_DIR=$(pwd)
cd build/scripts
./aws_build_dcp_from_cl.py

# 5. 创建 AFI
# 这会生成兼容的 DCP
```

## 当前推荐

### 短期方案（今天就要用）

**选项 A**: 切换到 us-west-2
```bash
# 1. 更新 launch_f1_vivado.sh
REGION="us-west-2"

# 2. 查找 us-west-2 的 AMI
aws ec2 describe-images \
  --owners 679593333241 \
  --filters "Name=name,Values=*FPGA*Ubuntu*" \
  --region us-west-2 \
  --query 'Images[*].[ImageId,Name]' \
  --output table

# 3. 更新 AMI ID
# 4. 重新运行
```

**选项 B**: 使用 F2 进行开发（不创建 AFI）
```bash
./run_fpga_flow.sh aws-launch f2
# 仅用于开发和测试
```

### 中期方案（本周）

1. **非高峰时段重试**
   - 晚上 10 PM - 早上 6 AM EST
   - 周末

2. **监控容量**
   ```bash
   # 每小时检查
   while true; do
     bash check_f1_availability.sh
     sleep 3600
   done
   ```

3. **提高 Spot 出价**
   ```bash
   # 在 launch_f1_vivado.sh 中
   SPOT_PRICE="1.00"  # 提高到 $1.00
   ```

### 长期方案（下周+）

1. **请求配额增加**
2. **考虑预留实例**
3. **使用多个区域**

## 成本对比

| 方案 | 区域 | 实例 | 成本/小时 | 4小时成本 |
|------|------|------|-----------|----------|
| F1 Spot | us-east-1 | f1.2xlarge | $0.50 | $2.00 |
| F1 按需 | us-east-1 | f1.2xlarge | $1.65 | $6.60 |
| F1 Spot | us-west-2 | f1.2xlarge | $0.50 | $2.00 |
| F2 Spot | us-east-1 | f2.6xlarge | $2.30 | $9.20 |

## 检查清单

- [ ] 运行 `check_f1_availability.sh` 检查配额
- [ ] 尝试 us-west-2 区域
- [ ] 尝试非高峰时段
- [ ] 考虑使用 F2 进行开发
- [ ] 如果紧急，使用按需实例（更贵但可靠）
- [ ] 请求配额增加（长期）

## 参考资料

- [AWS F1 实例](https://aws.amazon.com/ec2/instance-types/f1/)
- [AWS 区域和可用区](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/)
- [AWS Service Quotas](https://console.aws.amazon.com/servicequotas/)
- [AWS FPGA GitHub](https://github.com/aws/aws-fpga)

## 联系支持

如果问题持续，联系 AWS 支持：
- https://console.aws.amazon.com/support/home
- 说明你需要 F1 实例用于 FPGA 开发
- 提供你的使用场景和时间要求
