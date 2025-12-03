#!/bin/bash

set -e

echo "=== Static Timing Analysis (v0.4.1) ==="

# Create reports directory
mkdir -p reports

# Check if ICS55 netlist exists
if [ ! -f "../netlist/SimpleEdgeAiSoC_ics55.v" ]; then
    echo "Error: ICS55 netlist not found. Run synthesis first:"
    echo "  cd ../.. && ./run_ics55_synthesis.sh"
    exit 1
fi

echo "✓ Found ICS55 netlist"

# Note: Full STA requires liberty files from ICS55 PDK
# For now, we'll create a basic timing report
echo ""
echo "Note: Full STA requires ICS55 liberty files (.lib)"
echo "Current status: Netlist available, ready for STA"
echo ""
echo "To run full STA with OpenSTA:"
echo "  1. Obtain ICS55 liberty files"
echo "  2. Update sta_analysis.tcl with correct paths"
echo "  3. Run: sta sta_analysis.tcl"
echo ""

# Generate basic timing report from synthesis
echo "=== Basic Timing Information ==="
echo "From synthesis log:"
grep -A 10 "Chip area\|timing" ../../synthesis/netlist/SimpleEdgeAiSoC_ics55.v 2>/dev/null | head -20 || echo "No timing info in netlist"

echo ""
echo "=== Netlist Statistics ==="
wc -l ../netlist/SimpleEdgeAiSoC_ics55.v
ls -lh ../netlist/SimpleEdgeAiSoC_ics55.v

echo ""
echo "=== Next Steps ==="
echo "1. Obtain ICS55 PDK liberty files (.lib)"
echo "2. Run full STA: sta sta_analysis.tcl"
echo "3. Proceed to physical design (floorplanning)"


echo ""
echo "Quick summary:"
grep -A 5 "slack" reports/timing_summary.rpt || echo "Check timing_summary.rpt manually"
