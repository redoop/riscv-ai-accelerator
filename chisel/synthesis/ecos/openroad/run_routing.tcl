# OpenROAD Routing Script
# Continue from placement to complete routing

# Read libraries
read_lef tech_routing.lef

# Read design
read_def results/2_placement.def

# Read timing constraints
read_sdc ../sdc/timing.sdc

# Global routing
puts "\n=== Global Routing ==="
global_route \
    -guide_file results/route.guide \
    -verbose

# Detailed routing
puts "\n=== Detailed Routing ==="
detailed_route \
    -output_drc results/route.drc \
    -output_maze results/route.maze \
    -verbose

# Write results
write_def results/4_routing.def
write_verilog results/asic_top_routed.v

# Report statistics
report_checks -path_delay max
report_wns
report_tns
report_design_area

puts "\n=== Routing Complete ==="
puts "Output: results/4_routing.def"
