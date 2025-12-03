# OpenROAD 主配置文件

# PDK 路径
set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set IO_PATH "$PDK_ROOT/IP/IO/ICsprout_55LLULP1233_IO_251013"

# 设计文件
set DESIGN_NAME "asic_top"
set VERILOG_FILE "../project/netlist/asic_top_ics55.v"

# LEF 文件
set TECH_LEF "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"
set IO_LEF "$IO_PATH/lef/ICSIOA_N55_3P3_1P6M1TM.lef"

# Liberty 文件
set LIB_FILES [list \
    "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib" \
    "$IO_PATH/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib" \
]

# 时钟配置
set CLOCK_PORT "sys_clk_i_pad"
set CLOCK_PERIOD 10.0
set CLOCK_UNCERTAINTY 0.5

# Floorplan 配置
set DIE_AREA "0 0 600 600"
set CORE_AREA "50 50 550 550"
set CORE_UTILIZATION 0.7

# Placement 配置
set PLACE_DENSITY 0.7

# CTS 配置
set CTS_ROOT_BUF "BUFX8H7L"
set CTS_BUF_LIST "BUFX2H7L BUFX4H7L BUFX8H7L"
set CTS_WIRE_UNIT 20

# 输出目录
set RESULTS_DIR "./results"
set LOGS_DIR "./logs"
