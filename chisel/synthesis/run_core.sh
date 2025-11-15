#!/usr/bin/env bash
set -euo pipefail

PWD=$(cd "$(dirname "$0")";pwd)

# Use slang-friendly filelist
export FILELIST="${PWD}/filelist_core.f"
export TOP_NAME="SimpleEdgeAiSoC"
#export TOP_NAME="CoreMiniAxi"
export CLK_FREQ_MHZ="100"

export RESULT_DIR="${PWD}/netlist"
export NETLIST_FILE="${RESULT_DIR}/${TOP_NAME}_synth.v"
export TIMING_CELL_STAT_RPT="${RESULT_DIR}/timing_cell_stat.rpt"
export TIMING_CELL_COUNT_RPT="${RESULT_DIR}/timing_cell_count.rpt"
export GENERIC_STAT_JSON="${RESULT_DIR}/generic_stat.json"
export SYNTH_STAT_JSON="${RESULT_DIR}/synth_stat.json"
export SYNTH_CHECK_RPT="${RESULT_DIR}/synth_check.rpt"

export KEEP_HIERARCHY="false"
export CELL_DONT_USE=""
export CELL_TIE_LOW="TIELOH7R"
export CELL_TIE_LOW_PORT="Z"
export CELL_TIE_HIGH="TIEHIH7R"
export CELL_TIE_HIGH_PORT="Z"
export LIB_STDCELL="${PWD}/lib_ics55/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib ${PWD}/lib_ics55/ics55_LLSC_H7CR_ss_rcworst_1p08_125_nldm.lib"
export LIB_ALL=$LIB_STDCELL

# Launch yosys with the existing synthesis TCL
yosys ${PWD}/yosys/scripts/yosys_synthesis.tcl | tee ${PWD}/error.log || true
