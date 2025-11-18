# AWS AFI 创建成功！

## 问题分析与解决

### 原始错误
```
MANIFEST_NOT_FOUND: No manifest file was found.
```

### 根本原因

通过深入研究 AWS FPGA HDK 源代码（`aws_build_dcp_from_cl.py`），发现了 AWS 对 DCP tarball 的严格格式要求：

1. **目录结构**：tarball 必须包含 `to_aws/` 目录
2. **文件命名**：文件必须带时间戳前缀
   - DCP: `{timestamp}.SH_CL_routed.dcp`
   - Manifest: `{timestamp}.manifest.txt`
3. **Manifest 格式**：
   - 文件名：`manifest.txt`（不是 `manifest`）
   - Hash 算法：SHA256（不是 MD5）
   - 字段名：`pci_subsystem_id`（不是 `subsystem_id`）
   - Tool version：`v2024.1`（不是 `2024.1`）
   - 必须包含 clock recipe 字段

### 修复内容

#### 1. Tarball 结构
```bash
# 之前（错误）
tar -cvf file.tar SH_CL_routed.dcp manifest.txt

# 之后（正确）
mkdir to_aws/
cp file.dcp to_aws/20251118-045607.SH_CL_routed.dcp
cp manifest.txt to_aws/20251118-045607.manifest.txt
tar -cvf file.tar to_aws/
```

#### 2. Manifest 格式
```ini
# 正确的 manifest.txt 格式
manifest_format_version=2
pci_vendor_id=0x1D0F
pci_device_id=0xF000
pci_subsystem_id=0x1D51
pci_subsystem_vendor_id=0xFEDD
dcp_hash=<SHA256>
shell_version=0x04261818
dcp_file_name=20251118-045607.SH_CL_routed.dcp
hdk_version=1.4.23
tool_version=v2024.1
date=25_11_18-045607
clock_recipe_a=A1
clock_recipe_b=B0
clock_recipe_c=C0
clock_recipe_hbm=H0
```

#### 3. S3 Bucket 优化
```bash
# 之前：每次创建新 bucket
s3://fpga-afi-20251118-042141/

# 之后：使用固定 bucket + 子目录
s3://riscv-fpga-afi/builds/20251118-045607/dcp/
s3://riscv-fpga-afi/builds/20251118-045607/logs/
```

## 当前状态

### ⚠️ 新问题：Vivado 版本不兼容

**错误**: `The checkpoint was created with 'Vivado v2025.1', and cannot be opened in this version (v2024.1)`

**原因**: AWS AFI 服务使用 Vivado 2024.1，但 DCP 文件是用 Vivado 2025.1 生成的

**解决方案**: 需要在 F2 实例上使用 Vivado 2024.1 重新生成 DCP

详见: [FIX_VIVADO_VERSION.md](./FIX_VIVADO_VERSION.md)

### 检查状态
```bash
aws ec2 describe-fpga-images \
  --fpga-image-ids afi-0332a894b4002d5b2 \
  --region us-east-1
```

### 预计时间
AFI 生成通常需要 **30-60 分钟**

## 使用方法

### 1. 创建 AFI
```bash
cd chisel/synthesis/fpga
./run_fpga_flow.sh aws-create-afi
```

### 2. 监控状态
```bash
# 查看状态
aws ec2 describe-fpga-images \
  --fpga-image-ids afi-0332a894b4002d5b2 \
  --region us-east-1 \
  --query 'FpgaImages[0].State'

# 查看日志
aws s3 ls s3://riscv-fpga-afi/builds/20251118-045607/logs/ \
  --recursive --region us-east-1
```

### 3. 加载到 F1 实例
```bash
# 清除旧 AFI
sudo fpga-clear-local-image -S 0

# 加载新 AFI
sudo fpga-load-local-image -S 0 -I agfi-0416c67e9b36eb92d

# 验证
sudo fpga-describe-local-image -S 0 -H
```

## 关键学习点

1. **AWS 格式严格**：必须完全遵循 AWS HDK 的格式要求
2. **源码是最好的文档**：官方文档不够详细时，查看源码
3. **迭代调试**：通过错误消息逐步定位问题
4. **S3 优化**：使用固定 bucket + 子目录结构更经济

## 参考资料

- AWS FPGA HDK: `/opt/github/riscv-ai-accelerator/chisel/synthesis/fpga/aws-fpga/`
- Build Script: `hdk/common/shell_stable/build/scripts/aws_build_dcp_from_cl.py`
- Manifest Spec: `hdk/docs/AFI_Manifest.md`
- AFI Guide: `hdk/docs/Amazon_FPGA_Images_Afis_Guide.md`
