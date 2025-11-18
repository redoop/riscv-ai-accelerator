# AWS FPGA 完整工作流程

## 概述

本文档描述了从 Chisel 代码到 AWS F1 实例运行的完整流程。

## 前置条件

- AWS 账号配置完成
- AWS CLI 已安装并配置
- SSH 密钥对已创建

## 完整流程

### 阶段 1：本地准备（5-10 分钟）

#### 1.1 生成 Verilog
```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
./run_fpga_flow.sh prepare
```

这会：
- 编译 Chisel 代码
- 生成 SystemVerilog 文件
- 验证语法

### 阶段 2：AWS F2 构建（2-4 小时）

#### 2.1 启动 F2 实例
```bash
./run_fpga_flow.sh aws-launch
```

这会：
- 启动 f2.6xlarge Spot 实例
- 使用 Vivado ML 2024.1 Developer AMI（ami-0cab7155a229fac40）
- 预装 Vivado 2024.1，无需额外配置
- 保存实例信息到 `.f2_instance_info`

#### 2.2 上传项目
```bash
./run_fpga_flow.sh aws-upload
```

上传内容：
- Verilog 源文件
- 构建脚本
- 约束文件

#### 2.3 启动 Vivado 构建
```bash
./run_fpga_flow.sh aws-build
```

**重要**：Vivado ML 2024.1 Developer AMI 已预配置正确版本！

验证 Vivado 版本（可选）：
```bash
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@<F2_IP>
vivado -version
# 应该显示：Vivado v2024.1 (64-bit)
```

如果需要手动设置环境：
```bash
# 在 F2 实例上
source /tools/Xilinx/Vivado/2024.1/settings64.sh
```

#### 2.4 监控构建
```bash
./run_fpga_flow.sh aws-monitor
```

预计时间：2-4 小时

构建完成后会生成：
- `SH_CL_routed.dcp` - 设计检查点
- 时序报告
- 资源利用率报告

### 阶段 3：下载 DCP（5 分钟）

#### 3.1 下载 DCP 文件
```bash
./run_fpga_flow.sh aws-download-dcp
```

文件位置：`build/checkpoints/to_aws/SH_CL_routed.dcp`

#### 3.2 验证 DCP（可选但推荐）
```bash
# 检查文件大小（应该 > 1MB）
ls -lh build/checkpoints/to_aws/SH_CL_routed.dcp

# 检查是否是有效的 zip 文件
file build/checkpoints/to_aws/SH_CL_routed.dcp

# 验证 Vivado 版本
unzip -p build/checkpoints/to_aws/SH_CL_routed.dcp dcp.xml | grep -i version
```

### 阶段 4：创建 AFI（30-60 分钟）

#### 4.1 创建 AFI
```bash
./run_fpga_flow.sh aws-create-afi
```

这会：
1. 创建正确格式的 tarball：
   ```
   to_aws/
   ├── 20251118-HHMMSS.SH_CL_routed.dcp
   └── 20251118-HHMMSS.manifest.txt
   ```

2. 上传到 S3：
   ```
   s3://riscv-fpga-afi/builds/20251118-HHMMSS/dcp/
   s3://riscv-fpga-afi/builds/20251118-HHMMSS/logs/
   ```

3. 提交 AFI 创建请求

4. 自动监控进度（30-60 分钟）

#### 4.2 检查 AFI 状态
```bash
# 查看最新的 AFI 信息
cat aws-deployment/output/afi_info_*.txt | tail -20

# 手动检查状态
aws ec2 describe-fpga-images \
  --fpga-image-ids <AFI_ID> \
  --region us-east-1 \
  --query 'FpgaImages[0].State'
```

状态说明：
- `pending` - 正在生成
- `available` - 可用 ✅
- `failed` - 失败 ❌

#### 4.3 如果失败，查看日志
```bash
# 列出日志文件
aws s3 ls s3://riscv-fpga-afi/builds/<TIMESTAMP>/logs/ \
  --recursive --region us-east-1

# 下载 Vivado 日志
aws s3 cp s3://riscv-fpga-afi/builds/<TIMESTAMP>/logs/afi-<ID>/*_vivado.log \
  - --region us-east-1
```

### 阶段 5：清理 F2 实例（立即）

**重要**：构建完成后立即清理以节省成本！

```bash
./run_fpga_flow.sh aws-cleanup
```

这会：
- 终止所有 F2 实例
- 取消 Spot 请求
- 清理本地实例信息

成本节省：
- F2 实例：~$1.50/小时
- 如果忘记关闭 24 小时：~$36

### 阶段 6：测试 AFI（10-30 分钟）

#### 6.1 启动 F1 测试实例
```bash
# 启动 f1.2xlarge 实例（按需或 Spot）
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type f1.2xlarge \
  --key-name your-key \
  --region us-east-1
```

#### 6.2 安装 FPGA 管理工具
```bash
# SSH 到 F1 实例
ssh -i ~/.ssh/your-key.pem ec2-user@<F1_IP>

# 克隆 AWS FPGA 仓库
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh
```

#### 6.3 加载 AFI
```bash
# 清除旧 AFI
sudo fpga-clear-local-image -S 0

# 加载新 AFI
sudo fpga-load-local-image -S 0 -I <AGFI_ID>

# 验证
sudo fpga-describe-local-image -S 0 -H
```

#### 6.4 运行测试
```bash
# 编译测试程序
cd your-test-directory
make

# 运行测试
./test_program
```

#### 6.5 清理 F1 实例
```bash
# 在本地
aws ec2 terminate-instances --instance-ids <F1_INSTANCE_ID>
```

## 成本估算

| 阶段 | 资源 | 时间 | 成本 |
|------|------|------|------|
| F2 构建 | f2.6xlarge Spot | 2-4 小时 | $3-6 |
| AFI 创建 | AWS 服务 | 30-60 分钟 | 免费 |
| F1 测试 | f1.2xlarge | 10-30 分钟 | $0.30-0.90 |
| S3 存储 | 几 MB | 持续 | <$0.01/月 |
| **总计** | | | **$3.30-6.90** |

## 常见问题

### Q1: Vivado 版本不匹配
**错误**: `The checkpoint was created with 'Vivado v2025.1'`

**解决**: 在 F2 实例上使用 Vivado 2024.1：
```bash
source /tools/Xilinx/Vivado/2024.1/settings64.sh
```

### Q2: AFI 创建失败 - MANIFEST_NOT_FOUND
**原因**: Tarball 格式不正确

**解决**: 使用更新后的 `create_afi.sh` 脚本（已修复）

### Q3: AFI 创建失败 - CLK_ILLEGAL
**原因**: 缺少 clock recipe 配置

**解决**: Manifest 中已包含默认 clock recipes（已修复）

### Q4: 构建时间过长
**优化**:
- 使用更大的 F2 实例（f2.4xlarge）
- 简化设计
- 优化约束

### Q5: 成本控制
**建议**:
- 使用 Spot 实例（节省 70%）
- 构建完成后立即清理
- 设置 AWS 预算警报

## 自动化脚本

### 完整自动化流程
```bash
# 一键完成所有步骤（除了测试）
./run_fpga_flow.sh aws
```

这会自动执行：
1. 启动 F2 实例
2. 生成 Verilog
3. 上传项目
4. 启动构建
5. 监控进度
6. 下载 DCP
7. 创建 AFI

**注意**: 仍需手动清理 F2 实例！

## 检查清单

### 构建前
- [ ] AWS 凭证已配置
- [ ] SSH 密钥已创建
- [ ] Chisel 代码已测试
- [ ] 约束文件已准备

### 构建中
- [ ] F2 实例使用 Vivado 2024.1
- [ ] 构建日志无严重错误
- [ ] 时序收敛

### 构建后
- [ ] DCP 文件已下载
- [ ] DCP 版本正确（2024.1）
- [ ] F2 实例已清理 ✅

### AFI 创建
- [ ] Tarball 格式正确
- [ ] Manifest 完整
- [ ] AFI 状态为 available

### 测试
- [ ] F1 实例已启动
- [ ] AFI 加载成功
- [ ] 功能测试通过
- [ ] F1 实例已清理 ✅

## 参考资料

- [AWS FPGA HDK](https://github.com/aws/aws-fpga)
- [F2 实例文档](https://aws.amazon.com/ec2/instance-types/f2/)
- [AFI 创建指南](./AFI_CREATION_SUCCESS.md)
- [Vivado 版本修复](./FIX_VIVADO_VERSION.md)
