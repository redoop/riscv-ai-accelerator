# L1 Cache Implementation Documentation

## Overview

This document describes the implementation of the L1 cache controllers for the RISC-V AI accelerator chip. The L1 cache subsystem consists of separate instruction and data caches with comprehensive coherency support, performance monitoring, and error handling.

## Architecture

### Cache Hierarchy
- **L1 Instruction Cache**: 32KB, 4-way set associative, 64-byte cache lines
- **L1 Data Cache**: 32KB, 8-way set associative, 64-byte cache lines
- **Unified Controller**: Manages both caches with a single interface

### Key Features
1. **MESI Coherency Protocol**: Full cache coherency support
2. **Write-back Policy**: For data cache with dirty bit tracking
3. **LRU Replacement**: Least Recently Used replacement policy
4. **Performance Counters**: Comprehensive performance monitoring
5. **Error Handling**: Robust error detection and recovery
6. **Burst Transfers**: Efficient AXI4 burst support

## Implementation Details

### L1 Instruction Cache (l1_icache.sv)

#### Features
- **Size**: 32KB total capacity
- **Associativity**: 4-way set associative
- **Line Size**: 64 bytes
- **Sets**: 128 sets (32KB / (4 ways × 64 bytes))
- **Replacement**: True LRU policy
- **Coherency**: MESI protocol with snoop support

#### Key Components
```systemverilog
// Cache arrays
logic [TAG_BITS-1:0]    tag_array [SETS-1:0][WAYS-1:0];
logic [511:0]           data_array [SETS-1:0][WAYS-1:0];
logic                   valid [SETS-1:0][WAYS-1:0];
logic [1:0]             mesi_state [SETS-1:0][WAYS-1:0];
```

#### State Machine
- **IDLE**: Ready for new requests
- **LOOKUP**: Checking cache for hit/miss
- **MISS_REQ**: Requesting data from L2
- **MISS_WAIT**: Waiting for L2 response
- **SNOOP_CHECK**: Handling coherency requests

### L1 Data Cache (l1_dcache.sv)

#### Features
- **Size**: 32KB total capacity
- **Associativity**: 8-way set associative
- **Line Size**: 64 bytes
- **Sets**: 64 sets (32KB / (8 ways × 64 bytes))
- **Replacement**: Pseudo-LRU policy for 8-way cache
- **Write Policy**: Write-back with dirty bit tracking
- **Coherency**: MESI protocol with snoop support

#### Key Components
```systemverilog
// Cache arrays
logic [TAG_BITS-1:0]    tag_array [SETS-1:0][WAYS-1:0];
logic [511:0]           data_array [SETS-1:0][WAYS-1:0];
logic                   valid [SETS-1:0][WAYS-1:0];
logic                   dirty [SETS-1:0][WAYS-1:0];
logic [1:0]             mesi_state [SETS-1:0][WAYS-1:0];
```

#### Pseudo-LRU Implementation
The 8-way data cache uses a pseudo-LRU tree structure:
```systemverilog
logic [6:0] lru_bits [SETS-1:0];  // 7 bits for 8-way pseudo-LRU tree
```

### Unified Cache Controller (cache_controller.sv)

#### Features
- **Unified Interface**: Single interface for both I-cache and D-cache
- **Instruction/Data Selection**: `cpu_is_instr` signal selects cache type
- **Performance Monitoring**: Comprehensive performance counters
- **Writeback Support**: Handles dirty line writebacks to L2

#### Interface Signals
```systemverilog
// CPU interface
input  logic [ADDR_WIDTH-1:0]   cpu_addr;
input  logic [DATA_WIDTH-1:0]   cpu_wdata;
output logic [DATA_WIDTH-1:0]   cpu_rdata;
input  logic                    cpu_req;
input  logic                    cpu_we;
input  logic [DATA_WIDTH/8-1:0] cpu_be;
input  logic                    cpu_is_instr;
output logic                    cpu_ready;
output logic                    cpu_hit;
```

## Cache Coherency

### MESI Protocol States
- **Modified (11)**: Cache line is dirty and exclusive
- **Exclusive (10)**: Cache line is clean and exclusive
- **Shared (01)**: Cache line is clean and may be shared
- **Invalid (00)**: Cache line is invalid

### Snoop Interface
```systemverilog
// Snoop interface
input  logic                    snoop_req;
input  logic [ADDR_WIDTH-1:0]   snoop_addr;
input  logic [2:0]              snoop_type;
output logic                    snoop_hit;
output logic                    snoop_dirty;
output logic [2:0]              snoop_resp;
```

### Snoop Operations
- **Read Request (001)**: Check for shared/exclusive data
- **Invalidate (010)**: Invalidate matching cache lines
- **Response Codes**: Hit/miss, clean/dirty status

## Performance Features

### Performance Counters
Both caches include comprehensive performance monitoring:

#### I-Cache Counters
- `perf_hits`: Number of cache hits
- `perf_misses`: Number of cache misses
- `perf_accesses`: Total access count
- `perf_snoop_hits`: Snoop hit count

#### D-Cache Counters
- `perf_hits`: Number of cache hits
- `perf_misses`: Number of cache misses
- `perf_reads`: Read operation count
- `perf_writes`: Write operation count
- `perf_writebacks`: Writeback count
- `perf_snoop_hits`: Snoop hit count

### Burst Transfer Support
- **Multi-beat Transfers**: Support for AXI burst transfers
- **Beat Counting**: Proper handling of multi-beat cache line fills
- **Error Handling**: Detection and handling of AXI errors

## Testing

### Test Coverage
The comprehensive test suite covers:

1. **Basic Functionality**
   - Cache hits and misses
   - Read and write operations
   - Instruction vs data cache selection

2. **Cache Coherency**
   - Snoop hit detection
   - MESI state transitions
   - Invalidation handling

3. **Replacement Policy**
   - LRU behavior verification
   - Cache capacity testing
   - Conflict miss handling

4. **Performance Testing**
   - Hit/miss latency measurement
   - Throughput testing
   - Performance counter validation

5. **Stress Testing**
   - Sequential access patterns
   - Random access patterns
   - Strided access patterns
   - Mixed instruction/data workloads

### Test Files
- `test_l1_cache.sv`: Basic functionality tests
- `test_l1_cache_comprehensive.sv`: Comprehensive test suite

## Usage Examples

### Basic Cache Access
```systemverilog
// Instruction fetch
cpu_addr = 64'h1000;
cpu_is_instr = 1'b1;
cpu_req = 1'b1;
cpu_we = 1'b0;
wait(cpu_ready);
instruction_data = cpu_rdata;

// Data read
cpu_addr = 64'h2000;
cpu_is_instr = 1'b0;
cpu_req = 1'b1;
cpu_we = 1'b0;
wait(cpu_ready);
read_data = cpu_rdata;

// Data write
cpu_addr = 64'h2000;
cpu_wdata = 64'hDEADBEEF;
cpu_be = 8'hFF;
cpu_is_instr = 1'b0;
cpu_req = 1'b1;
cpu_we = 1'b1;
wait(cpu_ready);
```

### Byte-Enable Write
```systemverilog
// Write only lower 4 bytes
cpu_addr = 64'h3000;
cpu_wdata = 64'h12345678;
cpu_be = 8'b00001111;
cpu_is_instr = 1'b0;
cpu_req = 1'b1;
cpu_we = 1'b1;
wait(cpu_ready);
```

## Performance Characteristics

### Expected Performance
- **Hit Latency**: 1-2 clock cycles
- **Miss Latency**: 10-20 clock cycles (depending on L2)
- **Hit Rate**: >95% for typical workloads
- **Bandwidth**: Up to 64 bytes per clock cycle

### Optimization Features
- **Critical Word First**: Requested word returned first
- **Write Combining**: Multiple writes to same line combined
- **Prefetch Support**: Ready for future prefetch integration
- **Low Power**: Clock gating and power management ready

## Integration Notes

### AXI4 Interface Requirements
- **Data Width**: 512 bits (matches cache line size)
- **Address Width**: 64 bits
- **Burst Support**: INCR bursts for cache line transfers
- **Error Handling**: RESP field monitoring

### System Integration
- **Clock Domain**: Single clock domain with L2 cache
- **Reset**: Synchronous reset with proper initialization
- **Power Management**: Ready for clock/power gating
- **Debug Support**: Performance counter access via CSRs

## Future Enhancements

### Planned Features
1. **Prefetching**: Hardware prefetch support
2. **ECC Protection**: Error correction code support
3. **Way Prediction**: Reduce hit latency
4. **Sub-blocking**: Partial cache line support
5. **Compression**: Cache line compression support

### Scalability
- **Configurable Size**: Parameterized cache sizes
- **Variable Associativity**: Configurable way count
- **Multiple Ports**: Multi-port cache support
- **Hierarchical Coherency**: Support for more cache levels

## Conclusion

The L1 cache implementation provides a robust, high-performance foundation for the RISC-V AI accelerator chip. With comprehensive coherency support, performance monitoring, and thorough testing, it meets all requirements for modern processor cache subsystems while providing excellent performance characteristics for AI workloads.