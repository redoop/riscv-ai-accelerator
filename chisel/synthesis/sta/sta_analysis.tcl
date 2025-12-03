# OpenSTA Timing Analysis Script

# Read netlist (includes standard cells)
read_verilog ../ecos/project/netlist/asic_top_ics55.v
read_verilog ../ecos/project/netlist/ics55_LLSC_H7CL.v

# Link design
link_design asic_top

# Read constraints
read_sdc ../ecos/sdc/timing.sdc

# Report clocks
report_checks -path_delay min_max -format full_clock_expanded > reports/clock_report.rpt

# Report timing summary
report_checks -path_delay max -format full > reports/timing_summary.rpt

# Report setup violations
report_checks -path_delay max -slack_max 0 > reports/setup_violations.rpt

# Report hold violations  
report_checks -path_delay min -slack_max 0 > reports/hold_violations.rpt

# Report design statistics
report_design_area > reports/area.rpt
report_power > reports/power.rpt

puts "STA analysis complete. Check reports/ directory."
