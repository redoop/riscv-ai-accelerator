#!/bin/bash

set -e

echo "=== Complete OpenROAD P&R Flow ==="
echo "Design: RISC-V AI Accelerator"
echo "Target: 25 MHz"
echo ""

# Create results directory
mkdir -p results logs

# Step 1: Check if placement exists
if [ ! -f "results/2_placement.def" ]; then
    echo "Error: Placement DEF not found. Run placement first."
    exit 1
fi

echo "✓ Placement DEF found"

# Step 2: Run routing
echo ""
echo "=== Running Routing ==="
openroad -exit run_routing.tcl 2>&1 | tee logs/routing.log

# Check if routing succeeded
if [ -f "results/4_routing.def" ]; then
    echo ""
    echo "✅ Routing completed successfully!"
    echo ""
    echo "Generated files:"
    ls -lh results/4_routing.def
    ls -lh results/asic_top_routed.v 2>/dev/null || true
    echo ""
    echo "Check logs/routing.log for details"
else
    echo ""
    echo "❌ Routing failed. Check logs/routing.log"
    exit 1
fi

echo ""
echo "=== P&R Flow Complete ==="
echo "Next steps:"
echo "  1. Review timing: grep -A 10 'slack' logs/routing.log"
echo "  2. Check DRC: cat results/route.drc"
echo "  3. View layout: openroad -gui results/4_routing.def"
