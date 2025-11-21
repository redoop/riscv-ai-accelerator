# AWS AFI 设备兼容性问题

## 问题描述

AFI 创建失败，错误信息：
```
ERROR: [Constraints 18-884] HDPRVerify-01: design check point in-memory-design 
is using device xcvu47p, yet design check point ../checkpoints/SH_CL_BB_routed.dcp 
is using device xcvu9p. Both check points must be implemented using the same device.
```

## 根本原因

**AWS AFI 服务只支持 xcvu9p 设备（F1 实例），不支持 xcvu47p 设备（F2 实例）**

- **F1 实例**（已退役）：使用 xcvu9p (Virtex UltraScale+ VU9P)
- **F2 实例**（当前可用）：使用 xcvu47p (Virtex UltraScale+ VU47P)
- **AWS Shell**：基于 xcvu9p 设计
- **AFI 服务**：只接受 xcvu9p 的 DCP 文件

## 当前状态

由于 AWS F1 实例已于 2024 年退役，我们面临以下困境：

1. ✅ F2 实例可用，可以运行 Vivado
2. ❌ F2 实例生成的 DCP 使用 xcvu47p 设备
3. ❌ AWS AFI 服务拒绝 xcvu47p 的 DCP
4. ❌ 无法在 F2 上构建兼容 AFI 的设计

## 解决方案

### 方案 1：在 F2 上使用 xcvu9p 设备构建（推荐尝试）

修改构建脚本，在 F2 实例上指定使用 xcvu9p 设备：


**步骤**：
1. 确保使用 `build_fpga.tcl`（已配置为 xcvu9p）
2. 不要使用 `build_fpga_f2.tcl`（配置为 xcvu47p）
3. 在 F2 实例上运行构建

**优点**：
- 可以使用 F2 的计算资源
- 生成的 DCP 兼容 AFI 服务

**缺点**：
- F2 实例没有物理 xcvu9p 芯片
- 可能存在工具链兼容性问题

### 方案 2：本地构建（如果有 Vivado 许可证）

在本地机器上使用 Vivado 构建，指定 xcvu9p 设备。

### 方案 3：放弃 AFI，使用 F2 本地测试

如果不需要 AFI 部署，可以：
- 在 F2 上使用 xcvu47p 构建
- 生成完整的比特流
- 在 F2 实例上直接加载测试

## 立即行动

### 检查当前使用的构建脚本

```bash
# 查看 F2 实例上的构建命令
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@<F2_IP>
cd ~/fpga-project
cat build_command.sh  # 或查看实际使用的命令
```

### 使用正确的构建脚本

确保使用 `build_fpga.tcl`（xcvu9p）而不是 `build_fpga_f2.tcl`（xcvu47p）：

```bash
# 在 F2 实例上
cd ~/fpga-project/scripts
vivado -mode batch -source build_fpga.tcl
```

### 验证 DCP 设备类型

```bash
# 下载 DCP 后检查
strings SH_CL_routed.dcp | grep -i "xcvu"
```

## 预期结果

如果使用 xcvu9p 构建成功，你应该看到：
- DCP 文件包含 xcvu9p 设备信息
- AFI 创建不再报告设备不匹配错误

## 风险提示

⚠️ **重要**：在 F2 实例（物理芯片是 xcvu47p）上为 xcvu9p 设备构建可能会遇到：
- 资源约束不匹配
- 引脚分配问题
- 工具链警告

建议先小规模测试，确认可行性。

## 参考

- AWS F1 开发者指南（已过时）
- Vivado 设备支持文档
- AWS AFI 创建 API 文档
