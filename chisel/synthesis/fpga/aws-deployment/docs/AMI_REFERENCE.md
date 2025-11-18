# AWS FPGA AMI 参考

## 当前使用的 AMI

### F1 实例
- **AMI ID**: `ami-092fc5deb8f3c0f7d`
- **名称**: FPGA Developer AMI (Ubuntu) 1.16.1
- **Vivado 版本**: 2024.1
- **支持实例**: f1.2xlarge, f1.4xlarge, f1.16xlarge
- **设备**: xcvu9p (F1 兼容)
- **用户名**: ubuntu

### F2 实例
- **AMI ID**: `ami-0cab7155a229fac40`
- **名称**: Vivado ML 2024.1 Developer AMI
- **Vivado 版本**: 2024.1
- **支持实例**: f2.2xlarge, f2.6xlarge, f2.48xlarge
- **设备**: xcvu47p (F2 专用)
- **用户名**: ubuntu

## 查找最新 AMI

### 方法 1：使用 AWS CLI

```bash
# 查找 F1 FPGA Developer AMI
aws ec2 describe-images \
  --owners 679593333241 \
  --filters "Name=name,Values=*FPGA*Ubuntu*" \
  --region us-east-1 \
  --query 'Images[*].[ImageId,Name,Description]' \
  --output table

# 查找特定 Vivado 版本
aws ec2 describe-images \
  --owners 679593333241 \
  --filters "Name=name,Values=*FPGA*Ubuntu*" \
  --region us-east-1 \
  --query 'Images[*].[ImageId,Name,Description]' \
  --output text | grep "2024.1"
```

### 方法 2：AWS Marketplace

访问 AWS Marketplace 搜索 "FPGA Developer AMI"

## AMI 版本历史

| AMI ID | 名称 | Vivado | 支持实例 | 发布日期 |
|--------|------|--------|----------|----------|
| ami-092fc5deb8f3c0f7d | FPGA Developer AMI (Ubuntu) 1.16.1 | 2024.1 | F1 | 2025-04 |
| ami-01198b89d80ebfdd2 | FPGA Developer AMI (Ubuntu) 1.17.0 | 2024.2 | F1 | 2025-05 |
| ami-0cab7155a229fac40 | Vivado ML 2024.1 Developer AMI | 2024.1 | F2 | 2024 |

## 验证 AMI

### 检查 AMI 详情
```bash
aws ec2 describe-images \
  --image-ids ami-092fc5deb8f3c0f7d \
  --region us-east-1 \
  --query 'Images[0].[ImageId,Name,Description]' \
  --output text
```

### 检查支持的实例类型
```bash
# 启动实例时会验证 AMI 是否支持该实例类型
aws ec2 run-instances \
  --image-id ami-092fc5deb8f3c0f7d \
  --instance-type f1.2xlarge \
  --dry-run
```

## 常见问题

### Q1: 为什么 F1 和 F2 使用不同的 AMI？

**A**: F1 和 F2 使用不同的 FPGA 设备：
- F1: xcvu9p (需要特定的工具链和驱动)
- F2: xcvu47p (需要不同的工具链)

### Q2: 可以在 F1 上使用 F2 的 AMI 吗？

**A**: 不可以。AMI 与实例类型绑定：
- `ami-0cab7155a229fac40` 只支持 F2
- `ami-092fc5deb8f3c0f7d` 只支持 F1

### Q3: 如何选择 Vivado 版本？

**A**: 对于 AFI 创建，使用 Vivado 2024.1：
- AWS AFI 服务当前支持 2024.1
- 使用其他版本可能导致兼容性问题

### Q4: AMI 包含什么？

**F1 FPGA Developer AMI** 包含：
- Ubuntu 20.04/22.04
- Vivado 2024.1
- AWS FPGA HDK
- AWS FPGA SDK
- 预配置的环境变量

**F2 Vivado ML AMI** 包含：
- Ubuntu
- Vivado ML 2024.1
- 机器学习工具链

## 更新 AMI

### 更新 F1 启动脚本
```bash
# 编辑 launch_f1_vivado.sh
vi aws-deployment/launch_f1_vivado.sh

# 修改 AMI_ID 行
AMI_ID="ami-092fc5deb8f3c0f7d"
```

### 更新 F2 启动脚本
```bash
# 编辑 launch_f2_vivado.sh
vi aws-deployment/launch_f2_vivado.sh

# 修改 AMI_ID 行
AMI_ID="ami-0cab7155a229fac40"
```

## 区域支持

当前配置使用 `us-east-1` 区域。如果使用其他区域，需要查找该区域的 AMI ID：

```bash
# 例如：us-west-2
aws ec2 describe-images \
  --owners 679593333241 \
  --filters "Name=name,Values=*FPGA*Ubuntu*" \
  --region us-west-2 \
  --query 'Images[*].[ImageId,Name]' \
  --output table
```

## 故障排除

### 错误：InvalidAMIID.NotFound

**原因**: AMI ID 在当前区域不存在

**解决**:
1. 确认使用正确的区域
2. 查找该区域的正确 AMI ID
3. 更新脚本中的 AMI_ID

### 错误：Unsupported instance type

**原因**: AMI 不支持该实例类型

**解决**:
- F1 实例使用 `ami-092fc5deb8f3c0f7d`
- F2 实例使用 `ami-0cab7155a229fac40`

## 参考链接

- [AWS FPGA Developer AMI](https://aws.amazon.com/marketplace/pp/prodview-gimv3gqbpe57k)
- [AWS FPGA GitHub](https://github.com/aws/aws-fpga)
- [F1 实例文档](https://aws.amazon.com/ec2/instance-types/f1/)
- [F2 实例文档](https://aws.amazon.com/ec2/instance-types/f2/)
