# OpenROAD CTS 实施指南

## 当前状态

✅ OpenROAD 已安装  
✅ PDK 文件齐全（LEF + Liberty）  
✅ 网表已生成  
⚠️ 完整 P&R 流程需要更多配置

## 为什么无法立即运行 CTS？

### CTS 的前提条件

```
1. Floorplan（布图规划）
   ↓ 定义芯片尺寸和核心区域
   
2. Placement（布局）
   ↓ 放置所有 96,087 个单元
   
3. CTS（时钟树综合）← 我们在这里
   ↓ 插入时钟缓冲器
   
4. Routing（布线）
   ↓ 连接所有单元
```

**问题**：我们的设计有 96,087 个单元，需要先完成布局才能进行 CTS。

## 两个选择

### 选择 1：使用 OpenROAD-flow-scripts（推荐）⭐

这是 OpenROAD 官方的完整流程脚本，自动化所有步骤。

#### 安装

```bash
# 克隆仓库
git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
cd OpenROAD-flow-scripts

# 构建
./build_openroad.sh --local

# 测试
make
```

#### 配置

创建配置文件 `designs/asic_top/config.mk`：

```makefile
export DESIGN_NAME = asic_top
export PLATFORM    = ics55

export VERILOG_FILES = /path/to/asic_top_ics55.v
export SDC_FILE      = /path/to/timing.sdc

export DIE_AREA    = 0 0 600 600
export CORE_AREA   = 50 50 550 550

export CLOCK_PERIOD = 10.0
```

#### 运行

```bash
make DESIGN_CONFIG=./designs/asic_top/config.mk
```

**预计时间**：2-4 小时（取决于机器性能）

### 选择 2：手动 OpenROAD 流程

如果你想学习每个步骤的细节。

#### 步骤 1：Floorplan

```tcl
# 读取文件
read_lef ...
read_liberty ...
read_verilog ...
link_design asic_top

# 创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

# 布图规划
initialize_floorplan \
  -utilization 70 \
  -aspect_ratio 1.0 \
  -core_space 50
```

#### 步骤 2：Placement

```tcl
# 全局布局
global_placement -density 0.7

# 详细布局
detailed_placement
```

#### 步骤 3：CTS

```tcl
# 时钟树综合
clock_tree_synthesis \
  -root_buf BUFX8H7L \
  -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L" \
  -wire_unit 20
```

#### 步骤 4：验证

```tcl
# 报告
report_clock_skew
report_checks -path_delay max
```

## 实际建议

### 短期（本周）✅

```bash
# 使用方案 1：25MHz
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
./solution_25mhz.sh
```

**原因**：
- 立即可用
- 验证功能
- 为 CTS 做准备

### 中期（1-2周）

```bash
# 学习 OpenROAD-flow-scripts
git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
cd OpenROAD-flow-scripts

# 运行示例
make DESIGN_CONFIG=./designs/gcd/config.mk

# 理解流程
```

**学习内容**：
- Floorplan 配置
- Placement 参数
- CTS 选项
- Routing 设置

### 长期（2-4周）

```bash
# 应用到项目
# 1. 创建配置文件
# 2. 运行完整流程
# 3. 验证 100MHz
```

## 为什么这么复杂？

### ASIC 设计流程的现实

```
逻辑综合（Yosys）        ← 已完成 ✅
  ↓ 生成网表
  
布图规划（Floorplan）    ← 需要配置
  ↓ 定义芯片尺寸
  
布局（Placement）        ← 需要时间（数小时）
  ↓ 放置 96,087 个单元
  
时钟树综合（CTS）        ← 目标
  ↓ 插入缓冲器
  
布线（Routing）          ← 需要时间（数小时）
  ↓ 连接所有单元
  
物理验证（DRC/LVS）      ← 最终验证
  ↓
  
GDSII 生成              ← 流片
```

**每个步骤都需要**：
- 正确的配置
- 足够的时间
- 专业的工具

## 快速对比

| 方案 | 时间 | 复杂度 | 结果 |
|------|------|--------|------|
| 方案 1（25MHz）| 立即 | ⭐ | 25 MHz ✅ |
| 方案 2（手动缓冲器）| 1天 | ⭐⭐ | ~60 MHz |
| OpenROAD-flow | 1周 | ⭐⭐⭐ | 100 MHz ✅ |
| 手动 OpenROAD | 2周 | ⭐⭐⭐⭐ | 100 MHz ✅ |

## 推荐路线

### 第 1 天：验证功能
```bash
./solution_25mhz.sh
```

### 第 1 周：学习工具
```bash
# 安装 OpenROAD-flow-scripts
# 运行示例设计
# 理解配置选项
```

### 第 2-3 周：应用到项目
```bash
# 创建配置
# 运行完整流程
# 调试问题
```

### 第 4 周：优化和验证
```bash
# 优化时序
# 达到 100MHz
# 准备流片
```

## 学习资源

### 官方文档
- OpenROAD: https://openroad.readthedocs.io/
- Flow Scripts: https://openroad-flow-scripts.readthedocs.io/

### 教程
- Getting Started: https://openroad.readthedocs.io/en/latest/user/GettingStarted.html
- YouTube: 搜索 "OpenROAD tutorial"

### 示例
- GCD 设计: `designs/gcd/`
- RISC-V 设计: `designs/ibex/`

## 总结

### 现实情况

✅ **可以达到 100MHz**：设计本身健康  
⚠️ **需要完整 P&R**：不是一键完成  
📚 **需要学习**：ASIC 设计是专业领域  

### 建议

1. **今天**：使用 25MHz 验证功能 ✅
2. **本周**：学习 OpenROAD-flow-scripts
3. **下月**：完成完整 P&R 流程

### 最终目标

```
当前：26 MHz（无 CTS）
  ↓
中期：25 MHz（方案 1）✅
  ↓
最终：100 MHz（完整 P&R）⭐
```

---

**下一步**：`./solution_25mhz.sh`

**创建时间**：2025-12-03 10:53
