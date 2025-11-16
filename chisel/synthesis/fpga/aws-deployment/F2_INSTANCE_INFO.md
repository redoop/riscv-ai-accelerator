# AWS F2 实例信息

**日期**：2025-11-16

## F2 vs F1 对比

### F2 实例（新一代）

| 实例类型 | vCPU | 内存 | FPGA | FPGA 内存 | Spot 价格 | 预估按需价格 |
|---------|------|------|------|----------|----------|------------|
| f2.6xlarge | 24 | 256 GB | 1x VU47P | 80 GB | $0.50-$0.72/小时 | ~$2.50/小时 |
| f2.12xlarge | 48 | 512 GB | 2x VU47P | 160 GB | - | ~$5.00/小时 |
| f2.48xlarge | 192 | 2 TB | 8x VU47P | 640 GB | - | ~$20.00/小时 |

### F1 实例（上一代）

| 实例类型 | vCPU | 内存 | FPGA | FPGA 内存 | 按需价格 |
|---------|------|------|------|----------|---------|
| f1.2xlarge | 8 | 122 GB | 1x VU9P | 64 GB | $1.65/小时 |
| f1.4xlarge | 16 | 244 GB | 2x VU9P | 128 GB | $3.30/小时 |
| f1.16xlarge | 64 | 976 GB | 8x VU9P | 512 GB | $13.20/小时 |

## FPGA 对比

### Virtex UltraScale+ VU47P (F2)
- **逻辑单元**：~2.8M
- **DSP Slices**：~9,000
- **Block RAM**：~2,500
- **UltraRAM**：~1,000
- **PCIe Gen4**：支持

### Virtex UltraScale+ VU9P (F1)
- **逻辑单元**：~2.5M
- **DSP Slices**：~6,840
- **Block RAM**：~2,160
- **UltraRAM**：~960
- **PCIe Gen3**：支持

## 可用性

### F2 实例
- ✅ us-east-1a, us-east-1b, us-east-1c, us-east-1d
- ✅ Spot 实例可用（$0.50-$0.72/小时）
- ⚠️ 最小实例：f2.6xlarge（比 F1 更大更贵）

### F1 实例
- ❌ us-east-1 当前无法启动（所有可用区）
- ⚠️ 可能是临时容量问题

## 成本对比

### 使用 F2.6xlarge（Spot 实例）

| 阶段 | 时间 | Spot 价格 | 按需价格 |
|------|------|----------|---------|
| 环境配置 | 15 分钟 | $0.18 | $0.63 |
| Vivado 构建 | 2-4 小时 | $1.00-$2.88 | $5.00-$10.00 |
| AFI 创建 | 30-60 分钟 | $0.25-$0.72 | $1.25-$2.50 |
| 测试验证 | 20 分钟 | $0.24 | $0.83 |
| **总计** | **3-5 小时** | **$1.67-$4.02** | **$7.71-$13.96** |

### 使用 F1.2xlarge（如果可用）

| 阶段 | 时间 | 成本 |
|------|------|------|
| 完整验证 | 3-5 小时 | $4.95-$8.25 |

## 推荐方案

### 方案 1：使用 F2 Spot 实例（推荐）

**优势**：
- ✅ 当前可用
- ✅ 成本更低（Spot 价格 $0.50-$0.72/小时）
- ✅ 更强大的 FPGA（VU47P）
- ✅ 更多资源（24 vCPU, 256 GB 内存）

**劣势**：
- ⚠️ Spot 实例可能被中断（但概率较低）
- ⚠️ 需要修改 AMI（F2 可能需要不同的 AMI）

**命令**：
```bash
# 使用 Spot 实例
aws ec2 request-spot-instances \
  --spot-price "1.00" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0cb1b6ae2ff99f8bf",
    "InstanceType": "f2.6xlarge",
    "KeyName": "fpga-dev-key",
    "SecurityGroupIds": ["sg-03d27449f82b54360"],
    "Placement": {"AvailabilityZone": "us-east-1b"}
  }' \
  --region us-east-1
```

### 方案 2：使用 F2 按需实例

**优势**：
- ✅ 不会被中断
- ✅ 当前可用

**劣势**：
- ⚠️ 成本较高（~$2.50/小时）

**命令**：
```bash
aws ec2 run-instances \
  --image-id ami-0cb1b6ae2ff99f8bf \
  --instance-type f2.6xlarge \
  --key-name fpga-dev-key \
  --security-group-ids sg-03d27449f82b54360 \
  --placement AvailabilityZone=us-east-1b \
  --region us-east-1
```

### 方案 3：等待 F1 可用

**优势**：
- ✅ 成本最低（$1.65/小时）
- ✅ 文档和脚本已准备好

**劣势**：
- ⚠️ 不确定何时可用
- ⚠️ 可能需要等待数小时或数天

## 注意事项

### F2 实例使用注意

1. **AMI 兼容性**
   - F2 可能需要不同的 FPGA Developer AMI
   - 需要验证 ami-0cb1b6ae2ff99f8bf 是否支持 F2

2. **工具链版本**
   - F2 可能需要更新的 Vivado 版本
   - VU47P 支持可能需要 Vivado 2022.1+

3. **Spot 实例风险**
   - Spot 实例可能被中断（但 FPGA 实例通常中断率很低）
   - 建议在非高峰时段使用
   - 可以设置最高价格避免意外高费用

## 下一步行动

### 推荐：尝试 F2 Spot 实例

1. **验证 AMI 兼容性**
   ```bash
   aws ec2 describe-images \
     --image-ids ami-0cb1b6ae2ff99f8bf \
     --region us-east-1 \
     --query 'Images[0].[Name,Description]'
   ```

2. **请求 Spot 实例**
   ```bash
   cd chisel/synthesis/fpga/aws-deployment
   ./launch_f2_spot.sh  # 新脚本
   ```

3. **如果成功，继续验证流程**

## 参考资料

- [AWS F2 实例文档](https://aws.amazon.com/ec2/instance-types/f2/)
- [Spot 实例最佳实践](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html)
- [Xilinx VU47P 数据手册](https://www.xilinx.com/products/silicon-devices/fpga/virtex-ultrascale-plus.html)

---

**更新时间**：2025-11-16 16:15  
**状态**：F2 实例可用，推荐使用 Spot 实例
