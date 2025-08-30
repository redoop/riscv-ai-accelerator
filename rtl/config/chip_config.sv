// Chip Configuration Header
// Global configuration parameters and definitions

`ifndef CHIP_CONFIG_SV
`define CHIP_CONFIG_SV

`timescale 1ns/1ps

// Import the configuration package
import chip_config_pkg::*;

// Global chip parameters
parameter XLEN = 64;
parameter DATA_WIDTH = 32;
parameter ADDR_WIDTH = 64;

// Core configuration
parameter NUM_CORES = 4;
parameter NUM_HARTS_PER_CORE = 1;

// Memory configuration
parameter HBM_CHANNELS = 8;
parameter HBM_DATA_WIDTH = 512;
parameter DRAM_ADDR_WIDTH = 34;

// AI accelerator configuration
parameter TPU_MAC_UNITS = 256;
parameter VPU_LANES = 16;
parameter AI_CACHE_SIZE = 1024; // KB

// NoC configuration
parameter NOC_MESH_SIZE_X = 4;
parameter NOC_MESH_SIZE_Y = 4;
parameter NOC_FLIT_WIDTH = 128;

// Power management
parameter NUM_POWER_DOMAINS = 8;
parameter DVFS_LEVELS = 4;

// PCIe configuration
parameter PCIE_LANES = 16;
parameter PCIE_GEN = 4;

// Clock frequencies (MHz)
parameter CPU_FREQ = 2000;
parameter AI_FREQ = 1500;
parameter NOC_FREQ = 1000;
parameter MEM_FREQ = 3200;

`endif // CHIP_CONFIG_SV