#!/bin/bash
# test_upload.sh - Simulate program upload for testing
# This script demonstrates what would happen during a real upload

PROG=${1:-hello_lcd}
BIN_FILE="build/${PROG}.bin"

echo "=== RISC-V AI SoC Program Upload Simulator ==="
echo ""

# Check if binary exists
if [ ! -f "$BIN_FILE" ]; then
    echo "❌ Error: Binary file not found: $BIN_FILE"
    echo "   Please run 'make $PROG' first"
    exit 1
fi

# Get file size
SIZE=$(wc -c < "$BIN_FILE")
echo "📦 Program: $PROG"
echo "📊 Size: $SIZE bytes"
echo ""

# Simulate upload process
echo "🔌 Connecting to device..."
sleep 0.5
echo "✅ Connected (simulated)"
echo ""

echo "📤 Uploading program..."
CHUNKS=$((SIZE / 256 + 1))
for i in $(seq 1 $CHUNKS); do
    PROGRESS=$((i * 100 / CHUNKS))
    printf "\rProgress: %3d%% [" $PROGRESS
    BARS=$((PROGRESS / 5))
    for j in $(seq 1 20); do
        if [ $j -le $BARS ]; then
            printf "="
        else
            printf " "
        fi
    done
    printf "]"
    sleep 0.02
done
echo ""
echo "✅ Upload complete!"
echo ""

echo "🚀 Running program..."
sleep 0.5
echo ""

# Simulate program output based on which program
case $PROG in
    hello_lcd)
        echo "=== Hello LCD Output ==="
        echo "UART initialized at 115200 bps"
        echo "LCD initialized"
        echo "Displaying: Hello RISC-V!"
        echo "Animation running..."
        echo "Heartbeat: . . . . ."
        ;;
    ai_demo)
        echo "=== AI Demo Output ==="
        echo "AI Demo Starting..."
        echo "Frame: Cat"
        echo "Frame: Dog"
        echo "Frame: Bird"
        echo "AI inference running at ~15 FPS"
        ;;
    benchmark)
        echo "=== Benchmark Output ==="
        echo "Performance Benchmark"
        echo "Testing UART..."
        echo "Testing LCD..."
        echo "Testing Graphics..."
        echo "Testing AI..."
        echo ""
        echo "=== Results ==="
        echo "UART: 11520 B/s"
        echo "LCD: 625K px/s"
        echo "Graphics: 15 FPS"
        echo "AI: 6 GOPS"
        ;;
    system_monitor)
        echo "=== System Monitor Output ==="
        echo "System Monitor Started"
        echo "CPU: 50 MHz"
        echo "Uptime: 0.0s"
        echo "UART RX: 0"
        echo "UART TX: 0"
        echo "AI: IDLE"
        echo "Monitoring..."
        ;;
    bootloader)
        echo "=== Bootloader Output ==="
        echo "RISC-V AI Bootloader v0.2"
        echo "Ready for commands..."
        echo ""
        echo "LCD Display:"
        echo "┌────────────────────┐"
        echo "│ RISC-V AI          │"
        echo "│ Bootloader         │"
        echo "│ v0.2               │"
        echo "│ Ready...           │"
        echo "└────────────────────┘"
        ;;
esac

echo ""
echo "✅ Program is running on device (simulated)"
echo ""
echo "💡 To run on real hardware:"
echo "   1. Connect RISC-V AI SoC via USB-UART"
echo "   2. Find device: ls /dev/tty* (macOS) or ls /dev/ttyUSB* (Linux)"
echo "   3. Run: make run PROG=$PROG PORT=/dev/ttyXXX"
echo ""
