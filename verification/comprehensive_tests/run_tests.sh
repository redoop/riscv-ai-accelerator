#!/bin/bash

# Comprehensive Test Runner Script
# Runs all test suites for RISC-V AI Accelerator project

set -e  # Exit on any error

# Check if we're running bash 4+ for associative arrays
if [ "${BASH_VERSION%%.*}" -lt 4 ]; then
    echo "Error: This script requires Bash 4.0 or later for associative arrays"
    echo "Current version: $BASH_VERSION"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Test suite configuration
declare -A TEST_SUITES=(
    ["comprehensive"]="Basic functionality and core features"
    ["ai_instructions"]="AI instruction set extensions"
    ["performance"]="Performance benchmarks and optimization"
    ["integration"]="System integration and multi-core coordination"
)

# Global test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
START_TIME=$(date +%s)

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_DIR/test_runner.log"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$LOG_DIR/test_runner.log"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$LOG_DIR/test_runner.log"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_DIR/test_runner.log"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for Verilator
    if ! command -v verilator &> /dev/null; then
        log_error "Verilator not found. Please install Verilator."
        exit 1
    fi
    
    # Check for make
    if ! command -v make &> /dev/null; then
        log_error "Make not found. Please install make."
        exit 1
    fi
    
    # Check for required RTL files
    local rtl_dir="$PROJECT_ROOT/rtl"
    if [ ! -d "$rtl_dir" ]; then
        log_error "RTL directory not found: $rtl_dir"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to setup test environment
setup_environment() {
    log_info "Setting up test environment..."
    
    # Clean previous results
    rm -f "$LOG_DIR"/*.log
    rm -f "$RESULTS_DIR"/*.xml
    rm -f "$SCRIPT_DIR"/*.vcd
    
    # Create missing RTL stubs if needed
    cd "$SCRIPT_DIR"
    if make create_stubs > "$LOG_DIR/stub_creation.log" 2>&1; then
        log_info "RTL stubs created successfully"
    else
        log_warning "Some issues creating RTL stubs (check log)"
    fi
    
    log_success "Environment setup completed"
}

# Function to run syntax check
run_syntax_check() {
    log_info "Running syntax check..."
    
    cd "$SCRIPT_DIR"
    if make syntax_check > "$LOG_DIR/syntax_check.log" 2>&1; then
        log_success "Syntax check passed"
        return 0
    else
        log_error "Syntax check failed (see $LOG_DIR/syntax_check.log)"
        return 1
    fi
}

# Function to run individual test suite
run_test_suite() {
    local test_name="$1"
    local test_description="$2"
    
    log_info "Running $test_name test suite: $test_description"
    
    local log_file="$LOG_DIR/${test_name}_test.log"
    local result_file="$RESULTS_DIR/${test_name}_results.txt"
    
    cd "$SCRIPT_DIR"
    
    # Run the test with timeout
    local timeout_duration=300  # 5 minutes
    
    if timeout $timeout_duration make "$test_name" > "$log_file" 2>&1; then
        # Parse results from log
        local pass_count=$(grep -o "Passed: [0-9]*" "$log_file" | tail -1 | grep -o "[0-9]*" || echo "0")
        local fail_count=$(grep -o "Failed: [0-9]*" "$log_file" | tail -1 | grep -o "[0-9]*" || echo "0")
        
        # Check if test actually ran (look for completion message)
        if grep -q "Test.*completed\|ALL.*TESTS.*PASSED\|TESTS.*FAILED" "$log_file"; then
            if [ "$fail_count" -eq 0 ] && [ "$pass_count" -gt 0 ]; then
                log_success "$test_name: $pass_count tests passed"
                PASSED_TESTS=$((PASSED_TESTS + 1))
                echo "PASS: $pass_count passed, $fail_count failed" > "$result_file"
            else
                log_error "$test_name: $fail_count tests failed, $pass_count passed"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                echo "FAIL: $pass_count passed, $fail_count failed" > "$result_file"
            fi
        else
            log_error "$test_name: Test did not complete properly"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo "ERROR: Test did not complete" > "$result_file"
        fi
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_error "$test_name: Test timed out after $timeout_duration seconds"
        else
            log_error "$test_name: Test failed with exit code $exit_code"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "ERROR: Test failed or timed out" > "$result_file"
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Function to generate test report
generate_report() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local report_file="$RESULTS_DIR/test_report.html"
    
    log_info "Generating test report..."
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>RISC-V AI Accelerator Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .test-suite { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RISC-V AI Accelerator Test Report</h1>
        <p class="timestamp">Generated on: $(date)</p>
        <p>Test Duration: ${duration} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Total Test Suites: $TOTAL_TESTS</p>
        <p class="pass">Passed: $PASSED_TESTS</p>
        <p class="fail">Failed: $FAILED_TESTS</p>
        <p>Success Rate: $(( TOTAL_TESTS > 0 ? (PASSED_TESTS * 100) / TOTAL_TESTS : 0 ))%</p>
    </div>
    
    <div class="details">
        <h2>Test Suite Details</h2>
EOF

    # Add details for each test suite
    for test_name in "${!TEST_SUITES[@]}"; do
        local result_file="$RESULTS_DIR/${test_name}_results.txt"
        local status="NOT RUN"
        local details=""
        
        if [ -f "$result_file" ]; then
            status=$(head -1 "$result_file" | cut -d: -f1)
            details=$(head -1 "$result_file" | cut -d: -f2-)
        fi
        
        local status_class="fail"
        if [ "$status" = "PASS" ]; then
            status_class="pass"
        fi
        
        cat >> "$report_file" << EOF
        <div class="test-suite">
            <h3>$test_name</h3>
            <p>Description: ${TEST_SUITES[$test_name]}</p>
            <p>Status: <span class="$status_class">$status</span></p>
            <p>Details: $details</p>
        </div>
EOF
    done
    
    cat >> "$report_file" << EOF
    </div>
    
    <div class="logs">
        <h2>Log Files</h2>
        <ul>
            <li><a href="../logs/test_runner.log">Main Test Runner Log</a></li>
            <li><a href="../logs/syntax_check.log">Syntax Check Log</a></li>
EOF

    for test_name in "${!TEST_SUITES[@]}"; do
        echo "            <li><a href=\"../logs/${test_name}_test.log\">$test_name Test Log</a></li>" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
        </ul>
    </div>
</body>
</html>
EOF
    
    log_success "Test report generated: $report_file"
}

# Function to print final summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "           TEST SUMMARY"
    echo "=========================================="
    echo "Total Test Suites: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Duration: $(($(date +%s) - START_TIME)) seconds"
    echo "=========================================="
    
    if [ $FAILED_TESTS -eq 0 ] && [ $TOTAL_TESTS -gt 0 ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
        return 0
    else
        echo -e "${RED}❌ $FAILED_TESTS TEST SUITE(S) FAILED${NC}"
        return 1
    fi
}

# Main execution function
main() {
    local run_mode="all"
    local specific_test=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                echo "Usage: $0 [OPTIONS] [TEST_SUITE]"
                echo ""
                echo "Options:"
                echo "  -h, --help          Show this help message"
                echo "  -s, --syntax-only   Run syntax check only"
                echo "  -q, --quick         Run quick tests (reduced simulation time)"
                echo "  -c, --clean         Clean build artifacts before running"
                echo ""
                echo "Test Suites:"
                for test_name in "${!TEST_SUITES[@]}"; do
                    echo "  $test_name: ${TEST_SUITES[$test_name]}"
                done
                echo ""
                echo "Examples:"
                echo "  $0                    # Run all test suites"
                echo "  $0 comprehensive     # Run only comprehensive tests"
                echo "  $0 -s                # Run syntax check only"
                echo "  $0 -q performance    # Run quick performance tests"
                exit 0
                ;;
            -s|--syntax-only)
                run_mode="syntax"
                shift
                ;;
            -q|--quick)
                export QUICK_TEST=1
                shift
                ;;
            -c|--clean)
                log_info "Cleaning build artifacts..."
                cd "$SCRIPT_DIR"
                make clean
                shift
                ;;
            *)
                if [[ -n "${TEST_SUITES[$1]}" ]]; then
                    run_mode="single"
                    specific_test="$1"
                else
                    log_error "Unknown test suite: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Initialize log
    echo "Test run started at $(date)" > "$LOG_DIR/test_runner.log"
    
    log_info "Starting RISC-V AI Accelerator test suite"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Test directory: $SCRIPT_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup environment
    setup_environment
    
    # Run tests based on mode
    case $run_mode in
        "syntax")
            if run_syntax_check; then
                log_success "Syntax check completed successfully"
                exit 0
            else
                log_error "Syntax check failed"
                exit 1
            fi
            ;;
        "single")
            log_info "Running single test suite: $specific_test"
            run_syntax_check || log_warning "Syntax check failed, continuing anyway"
            run_test_suite "$specific_test" "${TEST_SUITES[$specific_test]}"
            ;;
        "all")
            log_info "Running all test suites"
            run_syntax_check || log_warning "Syntax check failed, continuing anyway"
            
            # Run all test suites
            for test_name in "${!TEST_SUITES[@]}"; do
                run_test_suite "$test_name" "${TEST_SUITES[$test_name]}"
            done
            ;;
    esac
    
    # Generate report and summary
    generate_report
    
    if print_summary; then
        exit 0
    else
        exit 1
    fi
}

# Run main function with all arguments
main "$@"