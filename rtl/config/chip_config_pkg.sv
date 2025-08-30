// Chip Configuration Package
// Contains global parameters and type definitions

`timescale 1ns/1ps

package chip_config_pkg;

    // Data type enumeration
    typedef enum logic [2:0] {
        DATA_TYPE_INT8   = 3'b000,
        DATA_TYPE_INT16  = 3'b001,
        DATA_TYPE_INT32  = 3'b010,
        DATA_TYPE_FP16   = 3'b011,
        DATA_TYPE_FP32   = 3'b100,
        DATA_TYPE_FP64   = 3'b101
    } data_type_e;
    
    // Status codes
    parameter STATUS_OK = 8'h00;
    parameter STATUS_ERROR = 8'hFF;
    parameter STATUS_BUSY = 8'h01;
    
    // Cache parameters
    parameter CACHE_LINE_SIZE = 64;
    parameter L1_CACHE_SIZE = 32768;    // 32KB
    parameter L2_CACHE_SIZE = 1048576;  // 1MB
    parameter L3_CACHE_SIZE = 8388608;  // 8MB
    
    // AI accelerator parameters
    parameter TPU_ARRAY_SIZE = 64;
    parameter VPU_LANES = 16;
    parameter MAX_VECTOR_LENGTH = 512;
    
    // NoC parameters
    parameter NOC_FLIT_WIDTH = 32;
    parameter NOC_ADDR_WIDTH = 8;
    
    // Power management
    parameter NUM_VOLTAGE_LEVELS = 8;
    parameter NUM_FREQUENCY_LEVELS = 8;
    
endpackage