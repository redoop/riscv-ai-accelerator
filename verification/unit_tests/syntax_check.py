#!/usr/bin/env python3
"""
Simple syntax checker for SystemVerilog files
Performs basic syntax validation without requiring a simulator
"""

import os
import re
import sys

def check_systemverilog_syntax(filename):
    """Basic syntax checking for SystemVerilog files"""
    errors = []
    warnings = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            lines = content.split('\n')
    except FileNotFoundError:
        return [f"File not found: {filename}"], []
    
    # Basic syntax checks
    module_count = content.count('module ') - content.count('endmodule')
    if module_count != 0:
        errors.append(f"Unmatched module/endmodule pairs: {module_count}")
    
    # Check for basic SystemVerilog constructs
    if 'module ' in content and 'endmodule' not in content:
        errors.append("Module declaration without endmodule")
    
    # Check for balanced parentheses in each line
    for i, line in enumerate(lines, 1):
        paren_count = line.count('(') - line.count(')')
        if paren_count != 0:
            # This is just a warning as parentheses can span multiple lines
            pass
    
    # Check for common syntax issues
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.endswith(',;'):
            errors.append(f"Line {i}: Invalid syntax ',;'")
        if stripped.endswith(';;'):
            warnings.append(f"Line {i}: Double semicolon")
    
    return errors, warnings

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 syntax_check.py <file1.sv> [file2.sv] ...")
        sys.exit(1)
    
    total_errors = 0
    total_warnings = 0
    
    for filename in sys.argv[1:]:
        print(f"Checking {filename}...")
        errors, warnings = check_systemverilog_syntax(filename)
        
        if errors:
            print(f"  ERRORS ({len(errors)}):")
            for error in errors:
                print(f"    {error}")
            total_errors += len(errors)
        
        if warnings:
            print(f"  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"    {warning}")
            total_warnings += len(warnings)
        
        if not errors and not warnings:
            print(f"  ✓ No syntax issues found")
    
    print(f"\nSummary: {total_errors} errors, {total_warnings} warnings")
    
    if total_errors > 0:
        sys.exit(1)
    else:
        print("✓ All files passed basic syntax check")
        sys.exit(0)

if __name__ == "__main__":
    main()