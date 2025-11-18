# F1 实例构建流程

本目录包含使用 F1 实例构建 FPGA 设计并创建 AFI 的完整流程。

## 为什么使用 F1？

- ✅ **设备兼容**: xcvu9p 与 AWS AFI 服务完全兼容
- ✅ **成本更低**: Spot 价格约 $0.50/小时
- ✅ **支持 AFI**: 可以创建和测试 AFI
- ✅ **官方支持**: AWS FPGA HDK 官方推荐

## 快速开始

```bash
# 从项目根目录
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga

# 使用 F1 流程
./run_fpga_flow.sh aws-launch f1
```

## 完整流程

### 1. 启动 F1 实例
```bash
cd aws-deployment/f1
bash launch.sh
```

### 2. 上传项目
```bash
bash upload.sh
```

### 3. 启动构建
```bash
bash build.sh
```

### 4. 监控进度
```bash
bash monitor.sh
```

### 5. 下载 DCP
```bash
bash download_dcp.sh
```

### 6. 创建 AFI
```bash
bash create_afi.sh
```

### 7. 清理资源
```bash
bash cleanup.sh
```

## 文件说明

- `launch.sh` - 启动 F1 实例
- `upload.sh` - 上传项目到 F1
- `build.sh` - 在 F1 上启动 Vivado 构建
- `monitor.sh` - 监控构建进度
- `download_dcp.sh` - 下载生成的 DCP
- `create_afi.sh` - 创建 AFI
- `cleanup.sh` - 清理 F1 实例

## 成本估算

| 项目 | 时间 | 成本 (Spot) |
|------|------|-------------|
| F1 构建 | 2-4小时 | $1-2 |
| AFI 创建 | 30-60分钟 | 免费 |
| **总计** | | **$1-2** |

## 技术规格

- **实例类型**: f1.2xlarge
- **FPGA 设备**: xcvu9p (Virtex UltraScale+ VU9P)
- **逻辑单元**: 2.5M LUTs
- **Vivado 版本**: 2024.1
