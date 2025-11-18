# 设备不匹配问题分析

## 错误信息

```
ERROR: [Constraints 18-884] HDPRVerify-01: 
design check point in-memory-design is using device xcvu47p, 
yet design check point ../checkpoints/SH_CL_BB_routed.dcp is using device xcvu9p. 
Both check points must be implemented using the same device, package and speed grade
```

## 问题根源

### 设备不匹配

| 组件 | 设备 | 实例类型 | 说明 |
|------|------|----------|------|
| **你的 DCP** | xcvu47p | F2 | Virtex UltraScale+ VU47P |
| **AWS Shell** | xcvu9p | F1 | Virtex UltraScale+ VU9P |

### 为什么会出现这个问题？

1. **F2 vs F1 实例**
   - F2 实例使用较新的 VU47P FPGA
   - F1 实例使用较旧的 VU9P FPGA
   - AWS AFI 服务目前只支持 F1 (xcvu9p)

2. **构建环境错误**
   - 你在 F2 实例上构建了 DCP
   - 但 AFI 服务期望 F1 兼容的 DCP

## 解决方案

### 方案 1：使用 F1 实例构建（推荐）

AWS FPGA HDK 目前主要支持 F1 实例，需要在 F1 实例上构建。

#### 步骤：

1. **启动 F1 实例而不是 F2**
   ```bash
   # 修改 launch_f2_vivado.sh 为 launch_f1_vivado.sh
   INSTANCE_TYPE="f1.2xlarge"  # 而不是 f2.6xlarge
   ```

2. **使用正确的 AMI**
   - FPGA Developer AMI for F1: `ami-0c55b159cbfafe1f0`
   - 包含 Vivado 2024.1 和 F1 Shell

3. **重新构建 DCP**
   ```bash
   # 在 F1 实例上
   cd aws-fpga
   source hdk_setup.sh
   cd hdk/cl/examples/your_design
   export CL_DIR=$(pwd)
   cd build/scripts
   ./aws_build_dcp_from_cl.py -c your_design
   ```

### 方案 2：检查 AWS F2 支持状态

F2 实例是较新的，可能 AWS 还未完全支持 F2 的 AFI 创建流程。

#### 验证步骤：

1. **查看 AWS 文档**
   ```bash
   # 检查 F2 是否支持 AFI 创建
   aws ec2 describe-fpga-images --help
   ```

2. **联系 AWS 支持**
   - 询问 F2 (xcvu47p) AFI 支持时间表
   - 确认当前是否只支持 F1 (xcvu9p)

### 方案 3：使用 AWS HDK 官方流程

按照 AWS FPGA HDK 官方文档，使用 F1 实例：

```bash
# 1. 启动 F1 实例
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type f1.2xlarge \
  --key-name your-key \
  --region us-east-1

# 2. 在 F1 上构建
ssh -i ~/.ssh/your-key.pem ec2-user@<F1_IP>
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source hdk_setup.sh

# 3. 使用 CL 示例或创建自己的设计
cd hdk/cl/examples/cl_hello_world
export CL_DIR=$(pwd)
cd build/scripts
./aws_build_dcp_from_cl.py
```

## 当前状态

### 你的设置
- ✗ 使用 F2 实例 (xcvu47p)
- ✗ 生成了 F2 兼容的 DCP
- ✗ AFI 服务拒绝 F2 DCP

### 需要的设置
- ✓ 使用 F1 实例 (xcvu9p)
- ✓ 生成 F1 兼容的 DCP
- ✓ AFI 服务接受 F1 DCP

## 修改脚本

### 更新 `launch_f2_vivado.sh` → `launch_f1_vivado.sh`

```bash
#!/bin/bash
# 使用 F1 实例而不是 F2

INSTANCE_TYPE="f1.2xlarge"  # 改为 F1
AMI_ID="ami-0c55b159cbfafe1f0"  # F1 Developer AMI
SPOT_PRICE="1.65"  # F1 价格

# ... 其余配置相同
```

### 更新构建脚本

确保使用正确的 Shell 和设备：

```tcl
# 在 Vivado TCL 脚本中
set_part xcvu9p-flgb2104-2-i  # F1 设备
```

## 成本对比

| 实例类型 | 设备 | 按需价格 | Spot 价格 | 用途 |
|----------|------|----------|-----------|------|
| f1.2xlarge | xcvu9p | $1.65/hr | ~$0.50/hr | 构建 + 测试 |
| f1.4xlarge | xcvu9p | $3.30/hr | ~$1.00/hr | 更快构建 |
| f2.2xlarge | xcvu47p | $2.55/hr | ~$0.77/hr | ❌ 不支持 AFI |
| f2.6xlarge | xcvu47p | $7.65/hr | ~$2.30/hr | ❌ 不支持 AFI |

## 推荐行动

1. **立即停止 F2 实例**
   ```bash
   ./run_fpga_flow.sh aws-cleanup
   ```

2. **创建 F1 启动脚本**
   ```bash
   cp aws-deployment/launch_f2_vivado.sh aws-deployment/launch_f1_vivado.sh
   # 编辑文件，改为 f1.2xlarge
   ```

3. **使用 F1 重新构建**
   ```bash
   ./run_fpga_flow.sh aws-launch  # 使用新的 F1 脚本
   ./run_fpga_flow.sh prepare
   ./run_fpga_flow.sh aws-upload
   ./run_fpga_flow.sh aws-build
   ```

4. **验证设备**
   ```bash
   # 在 F1 实例上
   vivado -version
   # 应该显示支持 xcvu9p
   ```

## 参考资料

- [AWS F1 实例文档](https://aws.amazon.com/ec2/instance-types/f1/)
- [AWS FPGA HDK](https://github.com/aws/aws-fpga)
- [F1 Shell 规格](https://github.com/aws/aws-fpga/blob/master/hdk/docs/AWS_Shell_Interface_Specification.md)

## 总结

**问题**：使用了 F2 实例 (xcvu47p) 构建 DCP，但 AWS AFI 服务只支持 F1 (xcvu9p)

**解决**：必须使用 F1 实例重新构建 DCP

**成本**：F1 Spot 实例约 $0.50/小时，比 F2 更便宜且兼容
