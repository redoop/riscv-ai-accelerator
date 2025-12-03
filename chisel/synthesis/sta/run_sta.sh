#!/bin/bash

set -e

echo "=== Static Timing Analysis ==="

# Create reports directory
mkdir -p reports

# Check if netlist exists
if [ ! -f "../ecos/project/netlist/SimpleEdgeAiSoC_synth.v" ]; then
    echo "Error: Netlist not found. Run ECOS synthesis first."
    exit 1
fi

# Check if liberty file exists
if [ ! -f "../ecos/pdk/ics55/lib/ics55_tt_1v2_25c.lib" ]; then
    echo "Error: Liberty file not found."
    exit 1
fi

# Run OpenSTA
echo "Running OpenSTA..."
sta sta_analysis.tcl

echo ""
echo "=== Analysis Complete ==="
echo "Reports generated in reports/ directory:"
ls -lh reports/

echo ""
echo "Quick summary:"
grep -A 5 "slack" reports/timing_summary.rpt || echo "Check timing_summary.rpt manually"
