# OpenSTA Script for ICS55 55nm
# v0.4.1 Static Timing Analysis

# Set PDK paths
set PDK_ROOT "../../pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"

# Read Liberty file (typical corner)
puts "Reading Liberty file..."
read_liberty $PDK_ROOT/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# Read Verilog netlist
puts "Reading netlist..."
read_verilog ../netlist/SimpleEdgeAiSoC_ics55.v

# Link design
puts "Linking design..."
link_design ip1_SimpleEdgeAiSoC

# Create clock constraints
puts "Creating clock constraints..."

# Main clock 100 MHz (10 ns period)
create_clock -name clk -period 10.0 [get_ports clock]

# SPI clock 10 MHz (generated, 10x divider)
# Note: Simplified - actual implementation may vary
create_generated_clock -name spi_clk \
    -source [get_ports clock] \
    -divide_by 10 \
    [get_pins lcd/lcd/spiClkReg/Q]

# Input delays (20% of clock period)
set_input_delay -clock clk -max 2.0 [all_inputs]
set_input_delay -clock clk -min 0.5 [all_inputs]

# Output delays (20% of clock period)
set_output_delay -clock clk -max 2.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]

# Clock uncertainty (5% of period)
set_clock_uncertainty 0.5 [all_clocks]

# Clock transition
set_clock_transition 0.1 [all_clocks]

# Load capacitance
set_load 0.05 [all_outputs]

# Create reports directory
file mkdir reports

# Report timing
puts "\n=== Timing Analysis ==="
puts "Analyzing setup timing..."
report_checks -path_delay max -format full_clock_expanded -fields {slew cap input nets fanout} -digits 3 > reports/setup_timing.rpt

puts "Analyzing hold timing..."
report_checks -path_delay min -format full_clock_expanded -fields {slew cap input nets fanout} -digits 3 > reports/hold_timing.rpt

# Report slack
puts "\n=== Slack Summary ==="
report_worst_slack -max > reports/setup_slack.rpt
report_worst_slack -min > reports/hold_slack.rpt

puts "Setup WNS:"
report_worst_slack -max
puts "\nHold WNS:"
report_worst_slack -min

# Report TNS
puts "\n=== Total Negative Slack ==="
report_tns > reports/tns.rpt
report_tns

# Report clock skew
puts "\n=== Clock Skew ==="
report_clock_skew > reports/clock_skew.rpt
report_clock_skew

# Report design statistics
puts "\n=== Design Statistics ==="
report_design_area > reports/design_area.rpt
report_design_area

# Summary
puts "\n=== Analysis Complete ==="
puts "Reports generated in reports/ directory:"
puts "  - setup_timing.rpt: Setup timing paths"
puts "  - hold_timing.rpt: Hold timing paths"
puts "  - setup_slack.rpt: Setup slack summary"
puts "  - hold_slack.rpt: Hold slack summary"
puts "  - tns.rpt: Total negative slack"
puts "  - clock_skew.rpt: Clock skew analysis"
puts "  - design_area.rpt: Design area statistics"

exit
