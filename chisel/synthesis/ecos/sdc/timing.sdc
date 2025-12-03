# Timing Constraints for SimpleEdgeAiSoC

# Main clock: 100MHz (10ns period)
create_clock -name clk -period 10.0 [get_ports clock]

# SPI clock: 10MHz (100ns period)
create_generated_clock -name spi_clk -source [get_ports clock] -divide_by 10 [get_pins lcd/spi_clk]

# Input/output delays (assume 2ns)
set_input_delay -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]

# Clock uncertainty (jitter + skew)
set_clock_uncertainty 0.5 [get_clocks clk]
set_clock_uncertainty 0.5 [get_clocks spi_clk]

# Clock transition
set_clock_transition 0.1 [get_clocks clk]

# False paths between clock domains
set_false_path -from [get_clocks clk] -to [get_clocks spi_clk]
set_false_path -from [get_clocks spi_clk] -to [get_clocks clk]
