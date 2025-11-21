# 快速修复：AFI 设备不匹配错误

## 问题
AFI 创建失败：`xcvu47p` vs `xcvu9p` 设备不匹配

## 原因
使用了错误的构建脚本（`build_fpga_f2.tcl` 使用 xcvu47p）

## 解决方案

### 步骤 1：重新构建（使用正确的设备）

在 F2 实例上重新运行构建，使用 xcvu9p 设备：

```bash
# SSH 到 F2 实例
source chisel/synthesis/fpga/aws-deployment/.f2_instance_info
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}

# 进入项目目录
cd fpga-project/scripts

# 停止旧的构建（如果还在运行）
pkill -f vivado

# 清理旧的构建文件
rm -rf ../build/checkpoints/*

# 使用正确的脚本重新构建
vivado -mode batch -source build_fpga.tcl
```

### 步骤 2：监控构建

```bash
# 在另一个终端监控日志
tail -f fpga-project/build/logs/vivado_build.log
```

### 步骤 3：验证 DCP 设备

构建完成后，验证 DCP 使用的设备：

```bash
# 在 F2 实例上
strings fpga-project/build/checkpoints/to_aws/SH_CL_routed.dcp | grep -i "xcvu"
# 应该看到 xcvu9p，而不是 xcvu47p
```

### 步骤 4：下载并重新创建 AFI

```bash
# 在本地
cd chisel/synthesis/fpga
./run_fpga_flow.sh aws-download-dcp
./run_fpga_flow.sh aws-create-afi
```

## 自动化脚本已更新

`start_build.sh` 已更新为使用 `build_fpga.tcl`（xcvu9p），
下次运行时会自动使用正确的设备。

## 预计时间

- 重新构建：2-4 小时
- AFI 创建：30-60 分钟
- 总计：3-5 小时

## 成本

- F2 Spot 实例：$2-4（构建）
- AFI 创建：免费
