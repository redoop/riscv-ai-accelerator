# 时钟树综合（CTS）指南

## 当前状态

**问题**：时钟网络直接驱动 25,556 个触发器，导致 27.593 ns 延迟

**需要**：进行时钟树综合（CTS）来插入缓冲器和平衡时钟树

## CTS 流程

### 方案 1：使用 OpenROAD（推荐）

完整的 OpenROAD 流程需要：

1. **布图规划（Floorplanning）**
2. **布局（Placement）**
3. **时钟树综合（CTS）**
4. **布线（Routing）**

**完整脚本示例**：
```tcl
# 读取 LEF
read_lef .../ics55_LLSC_H7CL.lef
read_lef .../ICSIOA_N55_3P3_1P6M1TM.lef

# 读取 Liberty
read_liberty .../ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_liberty .../ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

# 创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

# 布图规划
initialize_floorplan \
  -die_area "0 0 1000 1000" \
  -core_area "50 50 950 950" \
  -site core7

# 布局
global_placement
detailed_placement

# 时钟树综合
clock_tree_synthesis \
  -root_buf BUFX4H7L \
  -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L" \
  -wire_unit 20

# 输出
write_verilog project/netlist/asic_top_cts.v
write_def project/netlist/asic_top_cts.def

# 报告
report_clock_skew
report_checks
```

### 方案 2：使用 iEDA（中文工具）

iEDA 是国产 EDA 工具，支持完整的数字后端流程。

**安装**：
```bash
git clone https://gitee.com/oscc-project/iEDA.git
cd iEDA
./build.sh
```

**使用**：
```bash
iEDA -script cts_flow.tcl
```

### 方案 3：手动插入时钟缓冲器（简化）

如果无法运行完整的 P&R 流程，可以手动修改网表：

**步骤**：
1. 识别时钟扇出过大的区域
2. 手动插入 BUFX4H7L 缓冲器
3. 将时钟分成多个分支
4. 每个分支驱动 ~100 个触发器

**示例修改**：
```verilog
// 原始（25,556 个触发器直接连接）
wire sys_clk;
DFFQX1H7L ff1 (.CK(sys_clk), ...);
DFFQX1H7L ff2 (.CK(sys_clk), ...);
// ... 25,554 more

// 修改后（添加时钟缓冲器）
wire sys_clk;
wire sys_clk_buf1, sys_clk_buf2, ..., sys_clk_buf256;

// 第一级缓冲器
BUFX8H7L clk_buf_root (.A(sys_clk), .Y(sys_clk_root));

// 第二级缓冲器（256 个，每个驱动 ~100 个 FF）
BUFX4H7L clk_buf1 (.A(sys_clk_root), .Y(sys_clk_buf1));
BUFX4H7L clk_buf2 (.A(sys_clk_root), .Y(sys_clk_buf2));
// ... 254 more

// 触发器连接到分支
DFFQX1H7L ff1 (.CK(sys_clk_buf1), ...);
DFFQX1H7L ff2 (.CK(sys_clk_buf1), ...);
// ... 98 more on buf1

DFFQX1H7L ff101 (.CK(sys_clk_buf2), ...);
// ... 99 more on buf2
```

## 当前限制

### 为什么无法立即运行 CTS

1. **需要物理信息**：
   - 单元的位置坐标
   - 布线资源
   - 时序信息

2. **需要完整的 P&R 流程**：
   - Floorplanning
   - Placement
   - CTS
   - Routing

3. **工具复杂性**：
   - OpenROAD 需要正确的配置
   - 需要学习曲线

## 替代方案：降低频率验证

在完成 CTS 之前，可以降低频率来验证设计：

**修改 SDC**：
```tcl
# 降低到 30MHz（33.33 ns 周期）
create_clock -name sys_clk -period 33.33 [get_ports sys_clk_i_pad]
```

**运行 STA**：
```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
sta -exit run_sta_30mhz.tcl
```

## 预期结果

### CTS 后的改进

**时钟网络延迟**：
- 当前：27.593 ns
- CTS 后：< 1 ns
- 改进：**27x**

**时序**：
- 逻辑延迟：~10 ns
- 时钟延迟：~0.5 ns
- 总延迟：~10.5 ns
- **可以达到 100MHz** ✅

### 时钟树结构

```
sys_clk_i_pad
    ↓
[PAD Buffer]
    ↓
sys_clk_root
    ↓
[Root Buffer BUFX8H7L]
    ↓
├─ [Buf1 BUFX4H7L] → 100 FFs
├─ [Buf2 BUFX4H7L] → 100 FFs
├─ [Buf3 BUFX4H7L] → 100 FFs
...
└─ [Buf256 BUFX4H7L] → 56 FFs

总计：256 个缓冲器，25,556 个触发器
```

## 下一步建议

### 短期（本周）

1. **降低频率验证**：
   - 测试 30MHz
   - 确认功能正确

2. **学习 OpenROAD**：
   - 阅读文档
   - 运行示例

### 中期（1-2 周）

1. **完成 P&R 流程**：
   - Floorplanning
   - Placement
   - CTS
   - Routing

2. **验证时序**：
   - 重新运行 STA
   - 确认达到 100MHz

### 长期（项目完成）

1. **物理验证**：
   - DRC（设计规则检查）
   - LVS（版图与原理图一致性）

2. **生成 GDSII**：
   - 准备流片

## 参考资源

### OpenROAD
- 官网：https://theopenroadproject.org/
- 文档：https://openroad.readthedocs.io/
- GitHub：https://github.com/The-OpenROAD-Project/OpenROAD

### iEDA
- 官网：https://ieda.oscc.cc/
- Gitee：https://gitee.com/oscc-project/iEDA
- 文档：https://ieda-docs.oscc.cc/

### 教程
- OpenROAD Flow Scripts：https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
- iEDA 教程：https://ieda-docs.oscc.cc/zh/tutorial/

## 总结

**当前状态**：
- ✅ 识别了问题（缺少时钟树）
- ✅ 理解了解决方案（CTS）
- ⚠️ 需要完整的 P&R 流程

**建议**：
1. 短期：降低频率验证功能
2. 中期：学习并运行 OpenROAD/iEDA
3. 长期：完成完整的 ASIC 设计流程

---

**创建时间**：2025-12-03 10:20
