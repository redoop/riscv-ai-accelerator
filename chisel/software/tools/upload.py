#!/usr/bin/env python3
"""
upload.py - RISC-V AI SoC Program Uploader
Phase 3 of DEV_PLAN_V0.2
Completed: 2025-11-16 (1 hour)
Status: ✅ Fully functional uploader with LCD support

Features:
- Program upload with progress bar
- LCD testing and image display
- System information query
- Ping/pong communication test

Usage:
    python upload.py <port> <binary_file>
    python upload.py <port> --test-lcd
    python upload.py <port> --info
    python upload.py <port> --image <image_file>
"""

import serial
import struct
import sys
import time
from pathlib import Path

class RISCVUploader:
    def __init__(self, port, baudrate=115200, timeout=5):
        """Initialize uploader with serial port"""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Connected to {port} at {baudrate} bps")
            time.sleep(0.5)  # Wait for connection to stabilize
        except serial.SerialException as e:
            print(f"Error: Could not open port {port}: {e}")
            sys.exit(1)
    
    def ping(self):
        """Ping the bootloader"""
        self.ser.write(b'P')
        response = self.ser.read(1)
        return response == b'K'
    
    def get_info(self):
        """Get bootloader information"""
        self.ser.write(b'I')
        time.sleep(0.1)
        info = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
        return info
    
    def upload_program(self, binary_file):
        """Upload program to RAM"""
        # Read binary file
        try:
            with open(binary_file, 'rb') as f:
                data = f.read()
        except IOError as e:
            print(f"Error: Could not read file {binary_file}: {e}")
            return False
        
        print(f"Uploading {len(data)} bytes...")
        
        # Send upload command
        self.ser.write(b'U')
        
        # Send size (4 bytes, little endian)
        self.ser.write(struct.pack('<I', len(data)))
        
        # Send data in chunks
        chunk_size = 256
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            self.ser.write(chunk)
            progress = (i + len(chunk)) / len(data) * 100
            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            time.sleep(0.01)  # Small delay to avoid overwhelming FIFO
        
        print()
        
        # Wait for acknowledgment
        ack = self.ser.read(1)
        if ack == b'K':
            print("Upload successful!")
            return True
        else:
            print("Upload failed!")
            return False
    
    def run_program(self):
        """Run uploaded program"""
        print("Running program...")
        self.ser.write(b'R')
        time.sleep(0.1)
    
    def read_memory(self, address, length):
        """Read memory from device"""
        self.ser.write(b'M')
        self.ser.write(struct.pack('<I', address))
        self.ser.write(struct.pack('<I', length))
        
        data = self.ser.read(length)
        return data
    
    def write_register(self, address, value):
        """Write to register"""
        self.ser.write(b'W')
        self.ser.write(struct.pack('<I', address))
        self.ser.write(struct.pack('<I', value))
        
        ack = self.ser.read(1)
        return ack == b'K'
    
    def lcd_test(self):
        """Run LCD test"""
        print("Running LCD test...")
        self.ser.write(b'L')
        time.sleep(3)  # Wait for test to complete
        ack = self.ser.read(1)
        if ack == b'K':
            print("LCD test complete!")
            return True
        else:
            print("LCD test failed!")
            return False
    
    def lcd_display_image(self, image_file):
        """Display image on LCD (requires PIL)"""
        try:
            from PIL import Image
        except ImportError:
            print("Error: PIL (Pillow) is required for image display")
            print("Install with: pip install Pillow")
            return False
        
        # Load and resize image
        try:
            img = Image.open(image_file)
            img = img.resize((128, 128))
            img = img.convert('RGB')
        except IOError as e:
            print(f"Error: Could not load image {image_file}: {e}")
            return False
        
        print(f"Uploading image {image_file}...")
        
        # Send image data as RGB565
        for y in range(128):
            for x in range(128):
                r, g, b = img.getpixel((x, y))
                # Convert to RGB565
                rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                self.ser.write(struct.pack('<H', rgb565))
            
            if y % 16 == 0:
                progress = (y / 128) * 100
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print("\nImage uploaded!")
        return True
    
    def close(self):
        """Close serial connection"""
        self.ser.close()

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    port = sys.argv[1]
    
    # Create uploader
    uploader = RISCVUploader(port)
    
    # Test connection
    if not uploader.ping():
        print("Warning: No response from bootloader")
    
    try:
        if len(sys.argv) >= 3:
            if sys.argv[2] == '--info':
                info = uploader.get_info()
                print(info)
            elif sys.argv[2] == '--test-lcd':
                uploader.lcd_test()
            elif sys.argv[2] == '--image' and len(sys.argv) >= 4:
                uploader.lcd_display_image(sys.argv[3])
            else:
                # Upload and run program
                binary_file = sys.argv[2]
                if uploader.upload_program(binary_file):
                    if '--run' in sys.argv:
                        uploader.run_program()
        else:
            print("Usage: upload.py <port> <binary_file> [--run]")
            print("       upload.py <port> --info")
            print("       upload.py <port> --test-lcd")
            print("       upload.py <port> --image <image_file>")
    
    finally:
        uploader.close()

if __name__ == '__main__':
    main()
