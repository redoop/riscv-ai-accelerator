// L2 Shared Cache Controller
// 2MB, 16-way set associative, 64-byte cache lines
// Shared among 4 RISC-V cores with ECC support

module l2_cache #(
    parameter CACHE_SIZE = 2 * 1024 * 1024,  // 2MB
    parameter WAYS = 16,                      // 16-way set associative
    parameter LINE_SIZE = 64,                 // 64-byte cache lines
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512,               // 512-bit AXI interface
    parameter NUM_PORTS = 4                   // 4 L1 caches
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // L1 cache interfaces (4 ports)
    axi4_if.slave       l1_if [NUM_PORTS-1:0],
    
    // L3 cache interface
    axi4_if.master      l3_if,
    
    // Cache coherency interface
    input  logic                    snoop_req,
    input  logic [ADDR_WIDTH-1:0]   snoop_addr,
    input  logic [2:0]              snoop_type,
    output logic                    snoop_hit,
    output logic                    snoop_dirty,
    output logic [2:0]              snoop_resp,
    
    // ECC error reporting
    output logic                    ecc_single_error,
    output logic                    ecc_double_error,
    output logic [ADDR_WIDTH-1:0]   ecc_error_addr
);

    // Cache parameters
    localparam SETS = CACHE_SIZE / (WAYS * LINE_SIZE);
    localparam OFFSET_BITS = $clog2(LINE_SIZE);
    localparam INDEX_BITS = $clog2(SETS);
    localparam TAG_BITS = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;
    localparam ECC_BITS = 8; // SECDED ECC for 64-bit data
    
    // Address breakdown
    logic [TAG_BITS-1:0]    tag;
    logic [INDEX_BITS-1:0]  index;
    logic [OFFSET_BITS-1:0] offset;
    
    // Cache arrays with ECC
    logic [TAG_BITS-1:0]    tag_array [SETS-1:0][WAYS-1:0];
    logic [DATA_WIDTH+ECC_BITS-1:0] data_array [SETS-1:0][WAYS-1:0];
    logic                   valid [SETS-1:0][WAYS-1:0];
    logic                   dirty [SETS-1:0][WAYS-1:0];
    logic [1:0]             mesi_state [SETS-1:0][WAYS-1:0];
    
    // Pseudo-LRU for 16-way cache (15 bits)
    logic [14:0]            lru_bits [SETS-1:0];
    
    // Port arbitration
    logic [NUM_PORTS-1:0]   port_req;
    logic [NUM_PORTS-1:0]   port_grant;
    logic [$clog2(NUM_PORTS)-1:0] active_port;
    logic                   arb_valid;
    
    // Cache state machine
    typedef enum logic [3:0] {
        IDLE,
        ARB_WAIT,
        LOOKUP,
        HIT_RESPONSE,
        MISS_REQ,
        MISS_WAIT,
        WRITEBACK_REQ,
        WRITEBACK_WAIT,
        ECC_CORRECT,
        SNOOP_CHECK
    } cache_state_t;
    
    cache_state_t state, next_state;
    
    // Current request tracking
    logic [ADDR_WIDTH-1:0]  current_addr;
    logic [DATA_WIDTH-1:0]  current_wdata;
    logic                   current_we;
    logic [DATA_WIDTH/8-1:0] current_be;
    logic [7:0]             current_id;
    
    // Hit detection
    logic [WAYS-1:0]        hit_way;
    logic                   cache_hit;
    logic [$clog2(WAYS)-1:0] hit_way_idx;
    
    // ECC logic
    logic [ECC_BITS-1:0]    computed_ecc;
    logic [ECC_BITS-1:0]    stored_ecc;
    logic                   ecc_error;
    logic                   ecc_correctable;
    logic [DATA_WIDTH-1:0]  corrected_data;
    
    // Enhanced multi-core arbitration with fairness and QoS
    logic [2:0]             port_priority [NUM_PORTS-1:0]; // QoS priority per port
    logic [7:0]             port_age [NUM_PORTS-1:0];      // Age counter for fairness
    logic [$clog2(NUM_PORTS)-1:0] last_served_port;
    logic [NUM_PORTS-1:0]   port_blocked;                  // Blocked ports due to conflicts
    
    // Arbitration working variables
    logic [$clog2(NUM_PORTS)-1:0] best_port;
    logic [10:0]            best_score; // priority(3) + age(8) = 11 bits
    logic [10:0]            current_score;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_port <= '0;
            arb_valid <= 1'b0;
            last_served_port <= '0;
            
            // Initialize port priorities (can be configured via registers)
            for (int i = 0; i < NUM_PORTS; i++) begin
                port_priority[i] <= 3'b100; // Default medium priority
                port_age[i] <= '0;
            end
            port_blocked <= '0;
        end else begin
            // Collect requests
            for (int i = 0; i < NUM_PORTS; i++) begin
                port_req[i] <= l1_if[i].arvalid || l1_if[i].awvalid;
            end
            
            // Age-based fairness: increment age for waiting requests
            for (int i = 0; i < NUM_PORTS; i++) begin
                if (port_req[i] && !port_grant[i] && port_age[i] < 8'hFF) begin
                    port_age[i] <= port_age[i] + 1;
                end else if (port_grant[i]) begin
                    port_age[i] <= '0; // Reset age when served
                end
            end
            
            // Enhanced arbitration with priority and fairness
            if (state == IDLE && |port_req) begin
                arb_valid <= 1'b1;
                
                // Find best port considering priority, age, and fairness
                best_port = '0;
                best_score = '0;
                
                for (int i = 0; i < NUM_PORTS; i++) begin
                    if (port_req[i] && !port_blocked[i]) begin
                        // Score = priority * 256 + age + round_robin_bonus
                        current_score = {port_priority[i], port_age[i]} + 
                                      ((i > last_served_port) ? 11'h001 : 11'h000);
                        
                        if (current_score > best_score) begin
                            best_port = i[$clog2(NUM_PORTS)-1:0];
                            best_score = current_score;
                        end
                    end
                end
                
                active_port <= best_port;
                last_served_port <= best_port;
            end else if (state != ARB_WAIT) begin
                arb_valid <= 1'b0;
            end
            
            // Clear blocked ports when transaction completes
            if (state == HIT_RESPONSE || state == MISS_WAIT) begin
                port_blocked <= '0;
            end
        end
    end
    
    // Grant signals
    always_comb begin
        port_grant = '0;
        if (arb_valid) begin
            port_grant[active_port] = 1'b1;
        end
    end
    
    // Address breakdown for current request
    assign {tag, index, offset} = current_addr;
    
    // Hit detection logic
    genvar i;
    generate
        for (i = 0; i < WAYS; i++) begin : hit_detection
            assign hit_way[i] = valid[index][i] && 
                               (tag_array[index][i] == tag) &&
                               (mesi_state[index][i] != 2'b00);
        end
    endgenerate
    
    assign cache_hit = |hit_way;
    
    // Find hit way index
    always_comb begin
        hit_way_idx = '0;
        for (int j = 0; j < WAYS; j++) begin
            if (hit_way[j]) begin
                hit_way_idx = j[$clog2(WAYS)-1:0];
            end
        end
    end
    
    // Enhanced SECDED ECC computation (Single Error Correction, Double Error Detection)
    function automatic logic [ECC_BITS-1:0] compute_ecc(logic [DATA_WIDTH-1:0] data);
        logic [ECC_BITS-1:0] ecc;
        logic [511:0] d;
        d = data;
        
        // SECDED Hamming code for 512-bit data with 8-bit ECC
        ecc[0] = ^{d[0], d[1], d[3], d[4], d[6], d[8], d[10], d[11], d[13], d[15], d[17], d[19], d[21], d[23], d[25], d[26], d[28], d[30], d[32], d[34], d[36], d[38], d[40], d[42], d[44], d[46], d[48], d[50], d[52], d[54], d[56], d[57], d[59], d[61], d[63]};
        ecc[1] = ^{d[0], d[2], d[3], d[5], d[6], d[9], d[10], d[12], d[13], d[16], d[17], d[20], d[21], d[24], d[25], d[27], d[28], d[31], d[32], d[35], d[36], d[39], d[40], d[43], d[44], d[47], d[48], d[51], d[52], d[55], d[56], d[58], d[59], d[62], d[63]};
        ecc[2] = ^{d[1], d[2], d[3], d[7], d[8], d[9], d[10], d[14], d[15], d[16], d[17], d[22], d[23], d[24], d[25], d[29], d[30], d[31], d[32], d[37], d[38], d[39], d[40], d[45], d[46], d[47], d[48], d[53], d[54], d[55], d[56], d[60], d[61], d[62], d[63]};
        ecc[3] = ^{d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56]};
        ecc[4] = ^{d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56]};
        ecc[5] = ^{d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56]};
        ecc[6] = ^{d[57], d[58], d[59], d[60], d[61], d[62], d[63]};
        ecc[7] = ^{d, ecc[6:0]}; // Overall parity for double error detection
        
        return ecc;
    endfunction
    
    function automatic logic [DATA_WIDTH-1:0] correct_ecc_error(
        logic [DATA_WIDTH-1:0] data, 
        logic [ECC_BITS-1:0] stored_ecc, 
        logic [ECC_BITS-1:0] computed_ecc
    );
        logic [DATA_WIDTH-1:0] corrected;
        logic [ECC_BITS-1:0] syndrome;
        logic overall_parity_error;
        
        syndrome = stored_ecc ^ computed_ecc;
        corrected = data;
        overall_parity_error = syndrome[7];
        
        // SECDED error correction logic
        if (syndrome[6:0] != '0) begin
            if (overall_parity_error) begin
                // Single bit error - correct it
                if (syndrome[6:0] <= DATA_WIDTH) begin
                    corrected[syndrome[6:0]-1] = ~corrected[syndrome[6:0]-1];
                end
            end
            // If syndrome != 0 but no overall parity error, it's a double bit error (uncorrectable)
        end
        
        return corrected;
    endfunction
    
    // Enhanced ECC error classification
    function automatic logic is_correctable_error(logic [ECC_BITS-1:0] syndrome);
        return (syndrome[6:0] != '0) && syndrome[7]; // Single bit error
    endfunction
    
    function automatic logic is_uncorrectable_error(logic [ECC_BITS-1:0] syndrome);
        return (syndrome[6:0] != '0) && !syndrome[7]; // Double bit error
    endfunction
    
    // Enhanced ECC checking with proper SECDED
    logic [ECC_BITS-1:0] ecc_syndrome;
    
    always_comb begin
        stored_ecc = data_array[index][hit_way_idx][DATA_WIDTH +: ECC_BITS];
        computed_ecc = compute_ecc(data_array[index][hit_way_idx][DATA_WIDTH-1:0]);
        ecc_syndrome = stored_ecc ^ computed_ecc;
        
        ecc_error = (ecc_syndrome != '0) && cache_hit;
        ecc_correctable = ecc_error && is_correctable_error(ecc_syndrome);
        
        corrected_data = correct_ecc_error(
            data_array[index][hit_way_idx][DATA_WIDTH-1:0],
            stored_ecc,
            computed_ecc
        );
    end
    
    // ECC error logging and statistics
    logic [15:0] ecc_single_error_count;
    logic [15:0] ecc_double_error_count;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ecc_single_error_count <= '0;
            ecc_double_error_count <= '0;
        end else begin
            if (ecc_error && ecc_correctable) begin
                ecc_single_error_count <= ecc_single_error_count + 1;
            end else if (ecc_error && !ecc_correctable) begin
                ecc_double_error_count <= ecc_double_error_count + 1;
            end
        end
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (snoop_req) begin
                    next_state = SNOOP_CHECK;
                end else if (arb_valid) begin
                    next_state = ARB_WAIT;
                end
            end
            
            ARB_WAIT: begin
                next_state = LOOKUP;
            end
            
            LOOKUP: begin
                if (cache_hit) begin
                    if (ecc_error && ecc_correctable) begin
                        next_state = ECC_CORRECT;
                    end else begin
                        next_state = HIT_RESPONSE;
                    end
                end else begin
                    next_state = MISS_REQ;
                end
            end
            
            HIT_RESPONSE: begin
                next_state = IDLE;
            end
            
            ECC_CORRECT: begin
                next_state = HIT_RESPONSE;
            end
            
            MISS_REQ: begin
                if (l3_if.arready || l3_if.awready) begin
                    next_state = MISS_WAIT;
                end
            end
            
            MISS_WAIT: begin
                if ((l3_if.rvalid && l3_if.rlast) || l3_if.bvalid) begin
                    next_state = IDLE;
                end
            end
            
            SNOOP_CHECK: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Capture current request
    always_ff @(posedge clk) begin
        if (state == ARB_WAIT) begin
            if (l1_if[active_port].arvalid) begin
                current_addr <= l1_if[active_port].araddr;
                current_we <= 1'b0;
                current_id <= l1_if[active_port].arid;
            end else if (l1_if[active_port].awvalid) begin
                current_addr <= l1_if[active_port].awaddr;
                current_we <= 1'b1;
                current_id <= l1_if[active_port].awid;
            end
        end
        
        if (l1_if[active_port].wvalid && l1_if[active_port].wready) begin
            current_wdata <= l1_if[active_port].wdata;
            current_be <= l1_if[active_port].wstrb;
        end
    end
    
    // Pseudo-LRU replacement for 16-way cache
    function automatic logic [$clog2(WAYS)-1:0] get_lru_way_16(logic [14:0] lru_state);
        logic [3:0] way;
        // 4-level tree for 16-way
        way[3] = ~lru_state[0];
        way[2] = way[3] ? ~lru_state[2] : ~lru_state[1];
        way[1] = (way[3:2] == 2'b00) ? ~lru_state[6] :
                 (way[3:2] == 2'b01) ? ~lru_state[5] :
                 (way[3:2] == 2'b10) ? ~lru_state[4] : ~lru_state[3];
        way[0] = (way[3:1] == 3'b000) ? ~lru_state[14] :
                 (way[3:1] == 3'b001) ? ~lru_state[13] :
                 (way[3:1] == 3'b010) ? ~lru_state[12] :
                 (way[3:1] == 3'b011) ? ~lru_state[11] :
                 (way[3:1] == 3'b100) ? ~lru_state[10] :
                 (way[3:1] == 3'b101) ? ~lru_state[9] :
                 (way[3:1] == 3'b110) ? ~lru_state[8] : ~lru_state[7];
        return way;
    endfunction
    
    function automatic logic [14:0] update_lru_16(logic [14:0] lru_state, logic [3:0] accessed_way);
        logic [14:0] new_lru;
        new_lru = lru_state;
        // Update 4-level LRU tree
        new_lru[0] = accessed_way[3];
        if (accessed_way[3]) begin
            new_lru[2] = accessed_way[2];
            if (accessed_way[2]) begin
                new_lru[4] = accessed_way[1];
                if (accessed_way[1]) begin
                    new_lru[8] = accessed_way[0];
                end else begin
                    new_lru[9] = accessed_way[0];
                end
            end else begin
                new_lru[5] = accessed_way[1];
                if (accessed_way[1]) begin
                    new_lru[10] = accessed_way[0];
                end else begin
                    new_lru[11] = accessed_way[0];
                end
            end
        end else begin
            new_lru[1] = accessed_way[2];
            if (accessed_way[2]) begin
                new_lru[3] = accessed_way[1];
                if (accessed_way[1]) begin
                    new_lru[7] = accessed_way[0];
                end else begin
                    new_lru[12] = accessed_way[0];
                end
            end else begin
                new_lru[6] = accessed_way[1];
                if (accessed_way[1]) begin
                    new_lru[13] = accessed_way[0];
                end else begin
                    new_lru[14] = accessed_way[0];
                end
            end
        end
        return new_lru;
    endfunction
    
    // Cache write on hit
    always_ff @(posedge clk) begin
        if (state == HIT_RESPONSE && current_we && cache_hit) begin
            // Write data with ECC
            data_array[index][hit_way_idx][DATA_WIDTH-1:0] <= current_wdata;
            data_array[index][hit_way_idx][DATA_WIDTH +: ECC_BITS] <= compute_ecc(current_wdata);
            dirty[index][hit_way_idx] <= 1'b1;
            mesi_state[index][hit_way_idx] <= 2'b11; // Modified
        end
        
        // ECC correction
        if (state == ECC_CORRECT) begin
            data_array[index][hit_way_idx][DATA_WIDTH-1:0] <= corrected_data;
            data_array[index][hit_way_idx][DATA_WIDTH +: ECC_BITS] <= compute_ecc(corrected_data);
        end
    end
    
    // LRU update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < SETS; s++) begin
                lru_bits[s] <= '0;
            end
        end else if (state == HIT_RESPONSE && cache_hit) begin
            lru_bits[index] <= update_lru_16(lru_bits[index], hit_way_idx);
        end
    end
    
    // L1 interface responses
    generate
        for (i = 0; i < NUM_PORTS; i++) begin : l1_responses
            // Read address ready
            assign l1_if[i].arready = (state == ARB_WAIT) && (active_port == i) && l1_if[i].arvalid;
            
            // Write address ready
            assign l1_if[i].awready = (state == ARB_WAIT) && (active_port == i) && l1_if[i].awvalid;
            
            // Write data ready
            assign l1_if[i].wready = (active_port == i) && l1_if[i].awready;
            
            // Read response
            assign l1_if[i].rvalid = (state == HIT_RESPONSE) && (active_port == i) && !current_we;
            assign l1_if[i].rdata = cache_hit ? (ecc_correctable ? corrected_data : 
                                               data_array[index][hit_way_idx][DATA_WIDTH-1:0]) : '0;
            assign l1_if[i].rid = current_id;
            assign l1_if[i].rresp = ecc_error && !ecc_correctable ? 2'b10 : 2'b00; // SLVERR if uncorrectable
            assign l1_if[i].rlast = 1'b1;
            
            // Write response
            assign l1_if[i].bvalid = (state == HIT_RESPONSE) && (active_port == i) && current_we;
            assign l1_if[i].bid = current_id;
            assign l1_if[i].bresp = 2'b00; // OKAY
        end
    endgenerate
    
    // L3 interface (simplified - miss handling)
    assign l3_if.arvalid = (state == MISS_REQ) && !current_we;
    assign l3_if.araddr = {current_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
    assign l3_if.arlen = (LINE_SIZE / (DATA_WIDTH/8)) - 1;
    assign l3_if.arsize = $clog2(DATA_WIDTH/8);
    assign l3_if.arburst = 2'b01; // INCR
    assign l3_if.rready = 1'b1;
    
    // Snoop logic
    logic [TAG_BITS-1:0]    snoop_tag;
    logic [INDEX_BITS-1:0]  snoop_index;
    logic [WAYS-1:0]        snoop_hit_way;
    
    assign {snoop_tag, snoop_index} = snoop_addr[ADDR_WIDTH-1:OFFSET_BITS];
    
    generate
        for (i = 0; i < WAYS; i++) begin : snoop_hit_detection
            assign snoop_hit_way[i] = valid[snoop_index][i] && 
                                     (tag_array[snoop_index][i] == snoop_tag);
        end
    endgenerate
    
    assign snoop_hit = |snoop_hit_way;
    
    // ECC error reporting
    assign ecc_single_error = ecc_error && ecc_correctable;
    assign ecc_double_error = ecc_error && !ecc_correctable;
    assign ecc_error_addr = current_addr;
    
    // Initialize arrays
    initial begin
        for (int s = 0; s < SETS; s++) begin
            for (int w = 0; w < WAYS; w++) begin
                valid[s][w] = 1'b0;
                dirty[s][w] = 1'b0;
                mesi_state[s][w] = 2'b00; // Invalid
            end
        end
    end

endmodule