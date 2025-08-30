/*
 * Frequency Controller
 * 
 * This module generates different clock frequencies for DVFS operation
 * using PLLs and clock dividers.
 */

module frequency_controller #(
    parameter NUM_FREQUENCY_LEVELS = 8
) (
    input  logic       ref_clk,        // Reference clock (e.g., 100MHz)
    input  logic       rst_n,
    
    // Control interface
    input  logic [2:0] frequency_level_req,
    input  logic       frequency_enable,
    
    // Generated clocks
    output logic       cpu_clk,
    output logic       ai_accel_clk,
    output logic       memory_clk,
    output logic       noc_clk,
    
    // Status
    output logic       frequency_ready,
    output logic [2:0] current_frequency_level,
    output logic       pll_locked
);

    // Frequency level mapping (in MHz, assuming 100MHz reference)
    localparam logic [15:0] FREQUENCY_LEVELS [0:7] = '{
        16'd200,   // Level 0: 200MHz (ultra-low power)
        16'd400,   // Level 1: 400MHz (low power)
        16'd600,   // Level 2: 600MHz (power save)
        16'd800,   // Level 3: 800MHz (balanced)
        16'd1000,  // Level 4: 1000MHz (nominal)
        16'd1200,  // Level 5: 1200MHz (performance)
        16'd1400,  // Level 6: 1400MHz (high performance)
        16'd1600   // Level 7: 1600MHz (maximum performance)
    };

    // PLL configuration registers
    logic [7:0] pll_multiplier;
    logic [3:0] pll_divider;
    logic [2:0] target_level;
    logic pll_config_valid;
    logic pll_reconfig_req;
    
    // Clock generation
    logic pll_clk;
    logic [3:0] cpu_div_counter;
    logic [3:0] ai_div_counter;
    logic [3:0] mem_div_counter;
    logic [3:0] noc_div_counter;
    
    // Clock divider ratios for different domains
    logic [3:0] cpu_div_ratio;
    logic [3:0] ai_div_ratio;
    logic [3:0] mem_div_ratio;
    logic [3:0] noc_div_ratio;

    // PLL reconfiguration state machine
    typedef enum logic [2:0] {
        FREQ_IDLE,
        FREQ_RECONFIG,
        FREQ_WAIT_LOCK,
        FREQ_STABILIZE
    } freq_state_t;

    freq_state_t freq_state;
    logic [15:0] stabilize_counter;

    // Calculate PLL parameters based on target frequency
    always_comb begin
        case (frequency_level_req)
            3'd0: begin // 200MHz
                pll_multiplier = 8'd8;   // 100MHz * 8 = 800MHz
                pll_divider = 4'd4;      // 800MHz / 4 = 200MHz
                cpu_div_ratio = 4'd1;    // CPU at full speed
                ai_div_ratio = 4'd1;     // AI accelerator at full speed
                mem_div_ratio = 4'd2;    // Memory at half speed
                noc_div_ratio = 4'd1;    // NoC at full speed
            end
            3'd1: begin // 400MHz
                pll_multiplier = 8'd8;   // 100MHz * 8 = 800MHz
                pll_divider = 4'd2;      // 800MHz / 2 = 400MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
            3'd2: begin // 600MHz
                pll_multiplier = 8'd12;  // 100MHz * 12 = 1200MHz
                pll_divider = 4'd2;      // 1200MHz / 2 = 600MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
            3'd3: begin // 800MHz
                pll_multiplier = 8'd8;   // 100MHz * 8 = 800MHz
                pll_divider = 4'd1;      // 800MHz / 1 = 800MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
            3'd4: begin // 1000MHz (nominal)
                pll_multiplier = 8'd10;  // 100MHz * 10 = 1000MHz
                pll_divider = 4'd1;      // 1000MHz / 1 = 1000MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
            3'd5: begin // 1200MHz
                pll_multiplier = 8'd12;  // 100MHz * 12 = 1200MHz
                pll_divider = 4'd1;      // 1200MHz / 1 = 1200MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
            3'd6: begin // 1400MHz
                pll_multiplier = 8'd14;  // 100MHz * 14 = 1400MHz
                pll_divider = 4'd1;      // 1400MHz / 1 = 1400MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
            3'd7: begin // 1600MHz (maximum)
                pll_multiplier = 8'd16;  // 100MHz * 16 = 1600MHz
                pll_divider = 4'd1;      // 1600MHz / 1 = 1600MHz
                cpu_div_ratio = 4'd1;
                ai_div_ratio = 4'd1;
                mem_div_ratio = 4'd1;
                noc_div_ratio = 4'd1;
            end
        endcase
    end

    // Frequency control state machine
    always_ff @(posedge ref_clk or negedge rst_n) begin
        if (!rst_n) begin
            freq_state <= FREQ_IDLE;
            target_level <= 3'd4;  // Default to nominal frequency
            current_frequency_level <= 3'd4;
            frequency_ready <= 1'b0;
            pll_reconfig_req <= 1'b0;
            stabilize_counter <= '0;
        end else begin
            case (freq_state)
                FREQ_IDLE: begin
                    if (frequency_enable && (frequency_level_req != current_frequency_level)) begin
                        target_level <= frequency_level_req;
                        freq_state <= FREQ_RECONFIG;
                        frequency_ready <= 1'b0;
                        pll_reconfig_req <= 1'b1;
                    end
                end
                
                FREQ_RECONFIG: begin
                    pll_reconfig_req <= 1'b0;
                    freq_state <= FREQ_WAIT_LOCK;
                end
                
                FREQ_WAIT_LOCK: begin
                    if (pll_locked) begin
                        freq_state <= FREQ_STABILIZE;
                        stabilize_counter <= 16'd100;  // Wait for clock to stabilize
                    end
                end
                
                FREQ_STABILIZE: begin
                    if (stabilize_counter > 0) begin
                        stabilize_counter <= stabilize_counter - 1;
                    end else begin
                        current_frequency_level <= target_level;
                        frequency_ready <= 1'b1;
                        freq_state <= FREQ_IDLE;
                    end
                end
                
                default: freq_state <= FREQ_IDLE;
            endcase
        end
    end

    // Simplified PLL model (in real implementation, this would be a vendor-specific PLL)
    pll_model u_pll (
        .ref_clk(ref_clk),
        .rst_n(rst_n),
        .multiplier(pll_multiplier),
        .divider(pll_divider),
        .reconfig_req(pll_reconfig_req),
        .pll_clk(pll_clk),
        .locked(pll_locked)
    );

    // Clock dividers for different domains
    always_ff @(posedge pll_clk or negedge rst_n) begin
        if (!rst_n) begin
            cpu_div_counter <= '0;
            ai_div_counter <= '0;
            mem_div_counter <= '0;
            noc_div_counter <= '0;
            cpu_clk <= 1'b0;
            ai_accel_clk <= 1'b0;
            memory_clk <= 1'b0;
            noc_clk <= 1'b0;
        end else begin
            // CPU clock divider
            cpu_div_counter <= cpu_div_counter + 1;
            if (cpu_div_counter >= (cpu_div_ratio - 1)) begin
                cpu_div_counter <= '0;
                cpu_clk <= ~cpu_clk;
            end
            
            // AI accelerator clock divider
            ai_div_counter <= ai_div_counter + 1;
            if (ai_div_counter >= (ai_div_ratio - 1)) begin
                ai_div_counter <= '0;
                ai_accel_clk <= ~ai_accel_clk;
            end
            
            // Memory clock divider
            mem_div_counter <= mem_div_counter + 1;
            if (mem_div_counter >= (mem_div_ratio - 1)) begin
                mem_div_counter <= '0;
                memory_clk <= ~memory_clk;
            end
            
            // NoC clock divider
            noc_div_counter <= noc_div_counter + 1;
            if (noc_div_counter >= (noc_div_ratio - 1)) begin
                noc_div_counter <= '0;
                noc_clk <= ~noc_clk;
            end
        end
    end

endmodule

// Simplified PLL model
module pll_model (
    input  logic       ref_clk,
    input  logic       rst_n,
    input  logic [7:0] multiplier,
    input  logic [3:0] divider,
    input  logic       reconfig_req,
    output logic       pll_clk,
    output logic       locked
);

    logic [7:0] current_mult;
    logic [3:0] current_div;
    logic [15:0] lock_counter;
    logic reconfig_active;

    always_ff @(posedge ref_clk or negedge rst_n) begin
        if (!rst_n) begin
            current_mult <= 8'd10;  // Default multiplier
            current_div <= 4'd1;    // Default divider
            locked <= 1'b0;
            lock_counter <= '0;
            reconfig_active <= 1'b0;
        end else begin
            if (reconfig_req && !reconfig_active) begin
                current_mult <= multiplier;
                current_div <= divider;
                locked <= 1'b0;
                lock_counter <= '0;
                reconfig_active <= 1'b1;
            end else if (reconfig_active) begin
                if (lock_counter < 16'd1000) begin
                    lock_counter <= lock_counter + 1;
                end else begin
                    locked <= 1'b1;
                    reconfig_active <= 1'b0;
                end
            end
        end
    end

    // Simplified clock generation (in real implementation, this would be analog PLL)
    assign pll_clk = ref_clk;  // Simplified - actual PLL would generate scaled clock

endmodule