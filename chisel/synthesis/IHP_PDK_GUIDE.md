# IHP SG13G2 PDK 综合与仿真指南

## 概述

IHP SG13G2 是一个开源的 130nm BiCMOS 工艺开发套件（PDK），由德国 IHP 微电子研究所提供。与之前使用的 ICS55 工艺库不同，IHP PDK 提供了完整的 Verilog 行为模型，可以使用开源工具进行后综合仿真。

## 优势

✅ **完全开源**: Apache 2.0 许可证  
✅ **包含 Verilog 模型**: 可用 Icarus Verilog 仿真  
✅ **工业级工艺**: 130nm BiCMOS 技术  
✅ **完整的 PDK**: 包括标准单元、IO、SRAM 等  
✅ **活跃维护**: GitHub 上持续更新  

## 快速开始

### 1. 获取 PDK（已完成）

```bash
cd chisel/synthesis/pdk
python get_pdk.py
```

PDK 将被克隆到 `pdk/IHP-Open-PDK/` 目录。

### 2. 运行综合

使用 IHP PDK 进行逻辑综合：

```bash
cd chisel/synthesis
./run_ihp_synthesis.sh
```

这将：
- 使用 Yosys 读取 RTL 设计
- 映射到 IHP SG13G2 标准单元
- 生成网表: `netlist/SimpleEdgeAiSoC_ihp.v`
- 复制标准单元 Verilog 模型到 netlist 目录

### 3. 运行后综合仿真

使用 Icarus Verilog 进行仿真：

```bash
python run_post_syn_sim.py --simulator iverilog --netlist ihp
```

## 详细说明

### PDK 结构

```
pdk/IHP-Open-PDK/ihp-sg13g2/
├── libs.ref/
│   ├── sg13g2_stdcell/          # 标准单元库
│   │   ├── lib/                 # Liberty 时序库
│   │   │   ├── sg13g2_stdcell_typ_1p20V_25C.lib  # 典型工况
│   │   │   ├── sg13g2_stdcell_fast_1p32V_m40C.lib # 快速工况
│   │   │   └── sg13g2_stdcell_slow_1p08V_125C.lib # 慢速工况
│   │   └── verilog/
│   │       └── sg13g2_stdcell.v # Verilog 行为模型 ⭐
│   ├── sg13g2_io/               # IO 单元
│   └── sg13g2_sram/             # SRAM 宏单元
└── libs.tech/                   # 技术文件
```

### 标准单元库特点

IHP SG13G2 标准单元库包含：

- **基本逻辑门**: AND, OR, NAND, NOR, XOR, XNOR, INV, BUF
- **复杂门**: AOI, OAI, MUX 等
- **触发器**: DFF, DFFR, DFFS 等
- **锁存器**: DLATCH 等
- **特殊单元**: TIE, FILL, ANTENNA 等

单元命名规则：`sg13g2_<type>_<drive>`
- 例如: `sg13g2_inv_1` (反相器，驱动强度 1)
- 例如: `sg13g2_dfrbp_2` (带复位的 DFF，驱动强度 2)

### 综合脚本说明

`run_ihp_synthesis.sh` 执行以下步骤：

1. **检查 PDK**: 验证 Liberty 和 Verilog 文件存在
2. **读取 RTL**: 使用 Yosys 读取 SystemVerilog 设计
3. **综合优化**: 
   - `proc`: 处理过程块
   - `opt`: 优化
   - `fsm`: 有限状态机提取
   - `memory`: 内存映射
   - `techmap`: 技术映射
4. **单元映射**:
   - `dfflibmap`: 映射触发器到 IHP 单元
   - `abc`: 使用 ABC 进行组合逻辑优化和映射
5. **输出网表**: 生成 Verilog 网表

### 仿真流程

使用 Icarus Verilog 进行后综合仿真：

```bash
# 基本仿真
python run_post_syn_sim.py --simulator iverilog --netlist ihp

# 使用基本测试平台
python run_post_syn_sim.py --simulator iverilog --netlist ihp --testbench basic

# 查看波形
python run_post_syn_sim.py --wave --wave-tool gtkwave
```

仿真过程：
1. **编译**: iverilog 编译网表 + 标准单元模型 + 测试平台
2. **仿真**: vvp 运行编译后的仿真
3. **波形**: 生成 VCD 波形文件

## 与 ICS55 的对比

| 特性 | ICS55 | IHP SG13G2 |
|------|-------|------------|
| 工艺节点 | 55nm | 130nm |
| 开源程度 | 商业 | 完全开源 |
| Verilog 模型 | ❌ 不可用 | ✅ 包含 |
| 仿真工具 | VCS (商业) | Icarus Verilog (开源) |
| 许可证 | 专有 | Apache 2.0 |
| 文档 | 有限 | 完整 |

## 工作流程对比

### 之前（ICS55）

```
RTL → Yosys → ICS55 网表 → ❌ 无法仿真（缺少模型）
```

### 现在（IHP SG13G2）

```
RTL → Yosys → IHP 网表 → ✅ Icarus Verilog 仿真 → 波形分析
```

## 常见问题

### Q: 为什么选择 130nm 而不是更先进的工艺？

A: 
- IHP SG13G2 是目前最完整的开源 PDK 之一
- 130nm 对于学术研究和原型验证已经足够
- 完全开源，无需 NDA
- 有真实的流片服务支持

### Q: 性能会受影响吗？

A: 
- 130nm 比 55nm 慢，但对于验证功能足够
- 可以通过降低时钟频率来适配
- 主要用于功能验证，不是最终产品

### Q: 如何切换回 ICS55？

A: 
- 保留原有的综合脚本
- 使用不同的网表文件名
- 两种流程可以并存

### Q: 可以用于流片吗？

A: 
- 是的！IHP 提供 MPW (Multi-Project Wafer) 服务
- 通过 Europractice 或直接联系 IHP
- 完整的 DRC/LVS 支持

## 进阶使用

### 多工况分析

使用不同的 Liberty 文件进行多工况综合：

```bash
# 修改 run_ihp_synthesis.sh 中的 LIBERTY_FILE
# 快速工况（最好情况）
LIBERTY_FILE="$PDK_ROOT/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_fast_1p32V_m40C.lib"

# 慢速工况（最坏情况）
LIBERTY_FILE="$PDK_ROOT/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_slow_1p08V_125C.lib"
```

### 添加 IO 单元

如果需要 IO pad：

```bash
# 在综合脚本中添加 IO 库
IO_LIB="$PDK_ROOT/libs.ref/sg13g2_io/lib/sg13g2_io_typ_1p2V_3p3V_25C.lib"
IO_VERILOG="$PDK_ROOT/libs.ref/sg13g2_io/verilog/sg13g2_io.v"
```

### 使用 SRAM 宏单元

IHP PDK 包含多种 SRAM 配置：

```
- 64x32 (2-port)
- 512x64 (1-port)
- 1024x8 (1-port)
- 等等
```

## 资源链接

- **IHP PDK GitHub**: https://github.com/IHP-GmbH/IHP-Open-PDK
- **IHP 官网**: https://www.ihp-microelectronics.com/
- **文档**: https://github.com/IHP-GmbH/IHP-Open-PDK/tree/main/ihp-sg13g2/libs.doc
- **Yosys 文档**: https://yosyshq.net/yosys/
- **Icarus Verilog**: http://iverilog.icarus.com/

## 下一步

1. ✅ 已完成 PDK 集成
2. ✅ 已完成综合脚本
3. ✅ 已完成仿真支持
4. 🔄 运行综合和仿真验证
5. 📊 分析综合结果（面积、时序）
6. 🎯 优化设计以适配 130nm 工艺

## 总结

使用 IHP SG13G2 PDK 的主要优势：

1. **完全开源**: 无需 NDA，可以自由使用和分享
2. **可仿真**: 包含完整的 Verilog 模型
3. **工具链开源**: 使用 Yosys + Icarus Verilog
4. **真实工艺**: 可以实际流片
5. **社区支持**: 活跃的开源社区

这使得整个设计流程从 RTL 到后综合仿真都可以使用开源工具完成！
