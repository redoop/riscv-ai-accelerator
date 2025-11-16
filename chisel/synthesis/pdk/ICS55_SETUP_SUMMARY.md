# ICS55 PDK 设置总结

## ✅ 已完成的工作

### 1. PDK 安装
- ✅ ICS55 PDK 已下载到 `chisel/synthesis/pdk/icsprout55-pdk/`
- ✅ 使用的标准单元库：**ics55_LLSC_H7CL** (Low 功耗库)
- ✅ Liberty 文件：`ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib` (78MB)
- ✅ Verilog 模型：`ics55_LLSC_H7CL.v` (906KB)

### 2. 综合脚本
- ✅ 创建了 `run_ics55_synthesis.sh`
- ✅ 配置了正确的 PDK 路径
- ✅ 使用典型角度 (tt, 1.2V, 25°C)

### 3. 仿真支持
- ✅ 更新了 `run_post_syn_sim.py` 支持 ICS55 网表
- ✅ 添加了 `--netlist ics55` 选项
- ✅ 配置了正确的标准单元库路径

### 4. Makefile 集成
- ✅ 添加了 `synth_ics55` 目标
- ✅ 添加了 `sim_ics55` 目标
- ✅ 更新了 `info` 目标显示 PDK 状态

### 5. 文档
- ✅ 创建了 `ICS55_PDK_GUIDE.md` - 详细使用指南
- ✅ 创建了 `QUICK_START_ICS55.md` - 快速开始指南
- ✅ 更新了 `README.md` - 添加 ICS55 支持说明

## 📁 文件结构

```
chisel/synthesis/
├── pdk/
│   ├── get_ics55_pdk.py                    # PDK 下载脚本
│   └── icsprout55-pdk/                     # ICS55 PDK (已下载)
│       └── IP/STD_cell/ics55_LLSC_H7C_V1p10C100/
│           └── ics55_LLSC_H7CL/
│               ├── liberty/
│               │   └── ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
│               └── verilog/
│                   └── ics55_LLSC_H7CL.v
├── run_ics55_synthesis.sh                  # ICS55 综合脚本
├── run_post_syn_sim.py                     # 仿真脚本 (已更新)
├── Makefile                                # Makefile (已更新)
├── ICS55_PDK_GUIDE.md                      # 详细指南
├── QUICK_START_ICS55.md                    # 快速开始
└── README.md                               # 主文档 (已更新)
```

## 🚀 使用方法

### 方法 1: 使用脚本

```bash
# 1. 综合
cd chisel/synthesis
./run_ics55_synthesis.sh

# 2. 仿真
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

### 方法 2: 使用 Makefile

```bash
cd chisel/synthesis

# 综合
make synth_ics55

# 仿真
make sim_ics55

# 查看信息
make info
```

## 📊 PDK 对比

| 特性 | IHP SG13G2 | ICS55 |
|------|------------|-------|
| 工艺节点 | 130nm | 55nm |
| 标准单元库 | sg13g2_stdcell | ics55_LLSC_H7CL/H7CH/H7CR |
| Liberty 文件大小 | ~20MB | ~78MB |
| Verilog 模型大小 | ~500KB | ~906KB |
| 综合脚本 | `run_ihp_synthesis.sh` | `run_ics55_synthesis.sh` |
| 仿真选项 | `--netlist ihp` | `--netlist ics55` |

## 🔍 验证安装

```bash
# 检查 PDK 文件
ls -lh chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
ls -lh chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v

# 检查综合脚本
ls -lh chisel/synthesis/run_ics55_synthesis.sh

# 检查文档
ls -lh chisel/synthesis/ICS55_PDK_GUIDE.md
ls -lh chisel/synthesis/QUICK_START_ICS55.md
```

## 📝 配置详情

### 综合配置 (run_ics55_synthesis.sh)

```bash
PDK_ROOT="$SCRIPT_DIR/pdk/icsprout55-pdk"
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"
```

### 仿真配置 (run_post_syn_sim.py)

```python
elif netlist_type == "ics55":
    netlist = self.netlist_dir / f"{self.design_name}_ics55.v"
    stdcell_lib = self.netlist_dir / "ics55_LLSC_H7CL.v"
```

## 🎯 下一步

1. **生成 RTL** (如果还没有):
   ```bash
   cd chisel
   make verilog
   ```

2. **运行综合**:
   ```bash
   cd synthesis
   ./run_ics55_synthesis.sh
   ```

3. **运行仿真**:
   ```bash
   python run_post_syn_sim.py --simulator iverilog --netlist ics55
   ```

4. **查看结果**:
   ```bash
   cat netlist/synthesis_stats_ics55.txt
   ```

## 📚 文档链接

- [ICS55 PDK 详细指南](ICS55_PDK_GUIDE.md)
- [ICS55 快速开始](QUICK_START_ICS55.md)
- [IHP PDK 指南](IHP_PDK_GUIDE.md)
- [主 README](README.md)

## 💡 提示

1. **首次使用**: 建议先阅读 [QUICK_START_ICS55.md](QUICK_START_ICS55.md)
2. **详细配置**: 参考 [ICS55_PDK_GUIDE.md](ICS55_PDK_GUIDE.md)
3. **对比测试**: 可以同时使用 IHP 和 ICS55 PDK 进行对比
4. **标准单元库选择**: 
   - H7CL (Low): 低功耗，适合功耗敏感应用
   - H7CH (High): 高性能，适合性能关键路径
   - H7CR (Regular): 平衡，适合一般应用

## ⚠️ 注意事项

1. **文件大小**: ICS55 的 Liberty 文件较大 (78MB)，综合时间可能较长
2. **工艺角度**: 当前使用典型角度 (tt, 1.2V, 25°C)，可根据需要修改
3. **标准单元库**: 当前使用 H7CL (Low)，可根据需要切换到 H7CH 或 H7CR
4. **仿真时间**: 后端仿真比 RTL 仿真慢，这是正常的

## 🐛 故障排除

如果遇到问题，请参考：
- [ICS55_PDK_GUIDE.md](ICS55_PDK_GUIDE.md) 的"故障排除"章节
- [QUICK_START_ICS55.md](QUICK_START_ICS55.md) 的"常见问题"章节

---

**设置完成！** 🎉

现在你可以使用 ICS55 PDK 进行逻辑综合和后端仿真了。
