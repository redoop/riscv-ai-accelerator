# F1 实例信息文件验证

## 概述

所有 F1 启动脚本都会创建一个 `.f1_instance_info` 文件来保存实例信息。该文件对于后续操作（上传、构建、监控等）至关重要。

## 实例信息文件

### 文件位置

```
chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

### 文件内容

```bash
INSTANCE_ID=i-0123456789abcdef0
SPOT_REQUEST_ID=sir-xxxxxxxx  # 或 "on-demand"
PUBLIC_IP=54.123.45.67
KEY_NAME=fpga-f2-key
REGION=us-east-1
INSTANCE_TYPE=f1.2xlarge
AVAILABILITY_ZONE=us-east-1a
DEVICE=xcvu9p
BILLING_TYPE=spot  # 或 "on-demand"
TIMESTAMP="2024-11-18 15:30:00"
```

## 验证机制

### 自动验证

启动脚本会自动验证文件创建：

1. **文件存在性检查**
   ```bash
   if [ ! -f "$INFO_FILE" ]; then
       echo "❌ 实例信息文件创建失败"
       exit 1
   fi
   ```

2. **内容完整性检查**
   ```bash
   if ! grep -q "INSTANCE_ID=$INSTANCE_ID" "$INFO_FILE"; then
       echo "❌ 实例信息文件内容不完整"
       exit 1
   fi
   ```

### 手动验证

使用验证脚本检查实例信息：

```bash
cd chisel/synthesis/fpga/aws-deployment
./verify_instance_info.sh
```

**输出示例（成功）：**

```
=== F1 实例信息验证工具 ===

✓ 实例信息文件存在

✓ 所有必需字段都存在

=== 实例信息 ===
实例 ID:      i-0123456789abcdef0
实例类型:     f1.2xlarge
公网 IP:      54.123.45.67
可用区:       us-east-1a
区域:         us-east-1
FPGA 设备:    xcvu9p
密钥名称:     fpga-f2-key
计费类型:     on-demand
创建时间:     2024-11-18 15:30:00

=== 检查 AWS 实例状态 ===
✓ 实例状态: 运行中 (running)

=== 测试 SSH 连接 ===
测试 SSH 连接到 ubuntu@54.123.45.67...
✓ SSH 连接成功

=== 验证完成 ===
```

**输出示例（失败）：**

```
=== F1 实例信息验证工具 ===

❌ 实例信息文件不存在
文件路径: /path/to/.f1_instance_info

可能的原因:
  1. 尚未启动 F1 实例
  2. 实例启动失败
  3. 文件创建失败

启动 F1 实例:
  ./launch_f1_vivado.sh      # Spot 实例
  ./launch_f1_ondemand.sh    # 按需实例
  ./launch_fpga_instance.sh  # 交互式选择
```

## 常见问题

### 问题 1: 文件创建失败

**症状：**
```
❌ 实例信息文件创建失败
文件路径: /path/to/.f1_instance_info
```

**原因：**
1. 目录权限不足
2. 磁盘空间不足
3. 文件系统错误

**解决方案：**

1. 检查目录权限
```bash
ls -la chisel/synthesis/fpga/aws-deployment/
```

2. 检查磁盘空间
```bash
df -h
```

3. 手动创建文件测试
```bash
touch chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

4. 如果权限不足，修复权限
```bash
chmod 755 chisel/synthesis/fpga/aws-deployment/
```

### 问题 2: 文件内容不完整

**症状：**
```
❌ 实例信息文件内容不完整

文件已创建但内容可能损坏
```

**原因：**
1. 脚本在写入过程中被中断
2. 磁盘空间在写入时耗尽
3. 文件系统错误

**解决方案：**

1. 查看文件内容
```bash
cat chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

2. 删除损坏的文件
```bash
rm chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

3. 重新启动实例
```bash
./launch_f1_ondemand.sh
```

### 问题 3: 实例已创建但文件未保存

**症状：**
脚本显示实例创建成功，但退出时提示文件创建失败。

**影响：**
- 实例正在运行并产生费用
- 无法使用自动化脚本管理实例
- 需要手动记录实例信息

**解决方案：**

1. **立即记录实例信息**（脚本会显示）：
```
实例 ID: i-0123456789abcdef0
公网 IP: 54.123.45.67
可用区: us-east-1a
```

2. **手动创建信息文件**：
```bash
cat > chisel/synthesis/fpga/aws-deployment/.f1_instance_info << 'EOF'
INSTANCE_ID=i-0123456789abcdef0
SPOT_REQUEST_ID=on-demand
PUBLIC_IP=54.123.45.67
KEY_NAME=fpga-f2-key
REGION=us-east-1
INSTANCE_TYPE=f1.2xlarge
AVAILABILITY_ZONE=us-east-1a
DEVICE=xcvu9p
BILLING_TYPE=on-demand
TIMESTAMP="2024-11-18 15:30:00"
EOF
```

3. **验证文件**：
```bash
./verify_instance_info.sh
```

4. **或者终止实例并重新启动**：
```bash
# 终止实例
aws ec2 terminate-instances --instance-ids i-0123456789abcdef0 --region us-east-1

# 重新启动
./launch_f1_ondemand.sh
```

## 使用实例信息文件

### 后续操作依赖

以下操作都需要 `.f1_instance_info` 文件：

1. **上传项目**
```bash
./run_fpga_flow.sh aws-upload
```

2. **启动构建**
```bash
./run_fpga_flow.sh aws-build
```

3. **监控进度**
```bash
./run_fpga_flow.sh aws-monitor
```

4. **下载 DCP**
```bash
./run_fpga_flow.sh aws-download-dcp
```

5. **清理实例**
```bash
./run_fpga_flow.sh aws-cleanup
```

### 手动使用

你也可以在脚本中加载实例信息：

```bash
# 加载实例信息
source chisel/synthesis/fpga/aws-deployment/.f1_instance_info

# 使用变量
echo "实例 ID: $INSTANCE_ID"
echo "公网 IP: $PUBLIC_IP"

# SSH 连接
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}

# 查询实例状态
aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION
```

## 文件管理

### 查看文件

```bash
cat chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

### 备份文件

```bash
cp chisel/synthesis/fpga/aws-deployment/.f1_instance_info \
   chisel/synthesis/fpga/aws-deployment/.f1_instance_info.backup
```

### 删除文件

```bash
rm chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

**注意：** 删除文件不会终止实例，只是删除本地记录。

### 恢复文件

如果误删除文件但实例仍在运行：

1. 查找运行中的 F1 实例
```bash
aws ec2 describe-instances \
    --filters "Name=instance-type,Values=f1.2xlarge" \
              "Name=instance-state-name,Values=running" \
    --region us-east-1 \
    --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,Placement.AvailabilityZone]' \
    --output table
```

2. 手动重建信息文件（参考上面的格式）

## 安全建议

1. **不要提交到 Git**
   - `.f1_instance_info` 已在 `.gitignore` 中
   - 包含敏感信息（IP 地址、实例 ID）

2. **定期清理**
   - 实例终止后删除信息文件
   - 避免混淆旧实例信息

3. **权限控制**
   - 文件权限应为 600 或 644
   - 只有所有者可以修改

```bash
chmod 600 chisel/synthesis/fpga/aws-deployment/.f1_instance_info
```

## 故障排除流程

```
实例信息文件问题
    ↓
检查文件是否存在
    ↓
├─ 不存在 → 运行 verify_instance_info.sh
│              ↓
│          查看错误信息
│              ↓
│          按提示修复
│
└─ 存在 → 检查内容完整性
           ↓
       运行 verify_instance_info.sh
           ↓
       查看验证结果
           ↓
       ├─ 成功 → 继续后续操作
       └─ 失败 → 按提示修复
```

## 相关文档

- [F1 启动选项说明](./F1_LAUNCH_OPTIONS.md)
- [可用区自动选择](./AVAILABILITY_ZONE_SELECTION.md)
- [AWS 清理指南](./cleanup_fpga_instances.sh)
