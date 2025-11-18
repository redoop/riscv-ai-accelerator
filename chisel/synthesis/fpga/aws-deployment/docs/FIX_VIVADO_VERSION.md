# 修复 Vivado 版本不兼容问题

## 问题

```
ERROR: The checkpoint was created with 'Vivado v2025.1 (64-bit)', 
and cannot be opened in this version (v2024.1).
```

AWS AFI 服务使用 Vivado 2024.1，但你的 DCP 是用 Vivado 2025.1 生成的。

## 解决方案

### 方案 1：使用 Vivado 2024.1 重新构建（推荐）

#### 步骤 1：检查 F2 实例上的 Vivado 版本

```bash
# SSH 到 F2 实例
ssh -i ~/.ssh/your-key.pem ubuntu@<F2_IP>

# 检查 Vivado 版本
vivado -version
```

#### 步骤 2：如果是 2025.1，需要安装 2024.1

AWS FPGA Developer AMI 应该预装了正确的版本。如果没有：

```bash
# 下载 Vivado 2024.1
# 注意：需要 Xilinx 账号
wget https://www.xilinx.com/member/forms/download/xef.html?filename=Vivado_2024.1_...

# 或者使用 AWS 提供的版本
# 查看 AWS FPGA 文档获取正确的安装方法
```

#### 步骤 3：使用正确版本重新构建

```bash
# 在 F2 实例上
cd ~/fpga-project

# 确保使用 Vivado 2024.1
source /tools/Xilinx/Vivado/2024.1/settings64.sh

# 重新运行构建
vivado -version  # 确认是 2024.1
# 然后重新运行你的构建脚本
```

### 方案 2：检查 AWS 支持的 Vivado 版本

根据错误日志，AWS 当前支持的版本包括：
- v2018.2, v2018.3
- v2019.1, v2019.2
- v2020.1, v2020.2
- v2021.1, v2021.2
- v2022.1, v2022.2
- **v2024.1** ✅

### 方案 3：等待 AWS 支持 2025.1

如果你必须使用 2025.1 的特性，可以：
1. 联系 AWS 支持询问 2025.1 支持时间表
2. 提交 feature request

## 推荐操作流程

### 1. 启动正确的 F2 实例

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
./run_fpga_flow.sh aws-launch
```

确保使用的是 AWS FPGA Developer AMI，它应该预装 Vivado 2024.1。

### 2. 检查并设置正确的 Vivado 版本

在 F2 实例上：

```bash
# 列出可用的 Vivado 版本
ls /tools/Xilinx/Vivado/

# 使用 2024.1
source /tools/Xilinx/Vivado/2024.1/settings64.sh

# 验证
vivado -version
# 应该显示：Vivado v2024.1 (64-bit)
```

### 3. 重新构建 DCP

```bash
# 在 F2 实例上
cd ~/fpga-project

# 清理旧的构建
rm -rf build/checkpoints/*

# 重新构建
# 使用你的构建脚本或手动运行 Vivado
```

### 4. 下载新的 DCP 并创建 AFI

```bash
# 在本地
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
./run_fpga_flow.sh aws-download-dcp
./run_fpga_flow.sh aws-create-afi
```

## 验证 DCP 版本

在创建 AFI 之前，可以验证 DCP 的 Vivado 版本：

```bash
# 解压 DCP（它是一个 zip 文件）
unzip -l SH_CL_routed.dcp | grep -i version

# 或者在 Vivado 中检查
vivado -mode tcl
Vivado% open_checkpoint SH_CL_routed.dcp
Vivado% report_property [current_design]
```

## 更新 manifest.txt

确保 manifest.txt 中的 tool_version 匹配：

```ini
tool_version=v2024.1
```

## 参考

- AWS FPGA HDK 支持的 Vivado 版本：`aws-fpga/supported_vivado_versions.txt`
- AWS Developer AMI 文档：https://aws.amazon.com/marketplace/pp/prodview-gimv3gqbpe57k
