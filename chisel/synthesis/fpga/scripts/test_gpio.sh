#!/bin/bash
# Test GPIO functionality

set -e

echo "=== GPIO Test ==="

# Test pattern
TEST_PATTERN=0xA5A5A5A5

echo "Writing GPIO output: $(printf '0x%08x' $TEST_PATTERN)"

# Write to GPIO output register (address 0x2000)
sudo fpga-pci-write -s 0 -b 0 -a 0x2000 -d $(printf '0x%08x' $TEST_PATTERN)

sleep 0.1

# Read back from GPIO input register (address 0x2004)
READ_VALUE=$(sudo fpga-pci-read -s 0 -b 0 -a 0x2004)

echo "Reading GPIO input: $READ_VALUE"

if [ "$READ_VALUE" == "$(printf '0x%08x' $TEST_PATTERN)" ]; then
    echo "✓ GPIO read/write successful"
    echo "PASS: GPIO test"
else
    echo "ERROR: GPIO mismatch"
    echo "  Expected: $(printf '0x%08x' $TEST_PATTERN)"
    echo "  Read: $READ_VALUE"
    exit 1
fi

# Test GPIO direction control
echo "Testing GPIO direction control..."
sudo fpga-pci-write -s 0 -b 0 -a 0x2008 -d 0xFFFFFFFF  # All outputs
sleep 0.1

DIR_VALUE=$(sudo fpga-pci-read -s 0 -b 0 -a 0x2008)
if [ "$DIR_VALUE" == "0xffffffff" ]; then
    echo "✓ GPIO direction control successful"
else
    echo "WARNING: GPIO direction mismatch (got $DIR_VALUE)"
fi

echo "PASS: GPIO test completed"
