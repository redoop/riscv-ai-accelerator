#!/bin/bash
# RISC-V AI Accelerator UVM Test Runner
# Automated test execution and reporting script

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
REPORT_DIR="$SCRIPT_DIR/reports"
COV_DIR="$SCRIPT_DIR/coverage"

# Default settings
SIMULATOR="questa"
VERBOSITY="UVM_MEDIUM"
WAVES=1
COVERAGE=1
PARALLEL_JOBS=4
TIMEOUT=3600  # 1 hour timeout per test

# Test categories
SMOKE_TESTS=(
    "riscv_ai_smoke_test"
)

BASIC_TESTS=(
    "riscv_ai_random_test"
    "riscv_ai_matmul_test"
    "riscv_ai_conv2d_test"
    "riscv_ai_activation_test"
    "riscv_ai_memory_test"
)

ADVANCED_TESTS=(
    "riscv_ai_stress_test"
    "riscv_ai_power_test"
    "riscv_ai_error_test"
)

ALL_TESTS=("${SMOKE_TESTS[@]}" "${BASIC_TESTS[@]}" "${ADVANCED_TESTS[@]}")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_CATEGORY]

RISC-V AI Accelerator UVM Test Runner

OPTIONS:
    -s, --simulator SIMULATOR   Simulator to use (questa|vcs|xcelium) [default: questa]
    -v, --verbosity LEVEL       UVM verbosity level [default: UVM_MEDIUM]
    -w, --waves ENABLE          Enable waveform generation (0|1) [default: 1]
    -c, --coverage ENABLE       Enable coverage collection (0|1) [default: 1]
    -j, --jobs JOBS             Number of parallel jobs [default: 4]
    -t, --timeout SECONDS       Timeout per test in seconds [default: 3600]
    -o, --output-dir DIR        Output directory [default: current directory]
    -h, --help                  Show this help message

TEST_CATEGORY:
    smoke       Run smoke tests only
    basic       Run basic functionality tests
    advanced    Run advanced tests (stress, power, error)
    all         Run all tests [default]
    <test_name> Run specific test

EXAMPLES:
    $0                          # Run all tests with default settings
    $0 smoke                    # Run smoke tests only
    $0 -s vcs -j 8 basic       # Run basic tests with VCS using 8 parallel jobs
    $0 riscv_ai_matmul_test    # Run specific matrix multiplication test
    $0 --coverage 0 --waves 0  # Run without coverage or waves for faster execution

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--simulator)
                SIMULATOR="$2"
                shift 2
                ;;
            -v|--verbosity)
                VERBOSITY="$2"
                shift 2
                ;;
            -w|--waves)
                WAVES="$2"
                shift 2
                ;;
            -c|--coverage)
                COVERAGE="$2"
                shift 2
                ;;
            -j|--jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            smoke|basic|advanced|all)
                TEST_CATEGORY="$1"
                shift
                ;;
            riscv_ai_*)
                SPECIFIC_TEST="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$LOG_DIR" "$REPORT_DIR" "$COV_DIR"
    
    # Create timestamp for this run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_DIR="$REPORT_DIR/run_$TIMESTAMP"
    mkdir -p "$RUN_DIR"
    
    log_info "Run directory: $RUN_DIR"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if simulator is available
    case $SIMULATOR in
        questa)
            if ! command -v vsim &> /dev/null; then
                log_error "Questa/ModelSim not found. Please ensure it's in your PATH."
                exit 1
            fi
            ;;
        vcs)
            if ! command -v vcs &> /dev/null; then
                log_error "VCS not found. Please ensure it's in your PATH."
                exit 1
            fi
            ;;
        xcelium)
            if ! command -v xrun &> /dev/null; then
                log_error "Xcelium not found. Please ensure it's in your PATH."
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported simulator: $SIMULATOR"
            exit 1
            ;;
    esac
    
    # Check if UVM is available
    if [[ "$SIMULATOR" == "questa" ]]; then
        if ! vsim -c -do "quit" 2>&1 | grep -q "UVM"; then
            log_warn "UVM library may not be available. Tests may fail."
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Compile RTL and testbench
compile_design() {
    log_info "Compiling design..."
    
    cd "$SCRIPT_DIR"
    
    if make compile SIMULATOR="$SIMULATOR" > "$RUN_DIR/compile.log" 2>&1; then
        log_success "Compilation successful"
    else
        log_error "Compilation failed. Check $RUN_DIR/compile.log"
        exit 1
    fi
}

# Run a single test
run_single_test() {
    local test_name="$1"
    local test_log="$RUN_DIR/${test_name}.log"
    local test_start_time=$(date +%s)
    
    log_info "Running test: $test_name"
    
    cd "$SCRIPT_DIR"
    
    # Run test with timeout
    if timeout "$TIMEOUT" make run \
        TEST="$test_name" \
        SIMULATOR="$SIMULATOR" \
        VERBOSITY="$VERBOSITY" \
        WAVES="$WAVES" \
        COVERAGE="$COVERAGE" \
        > "$test_log" 2>&1; then
        
        local test_end_time=$(date +%s)
        local test_duration=$((test_end_time - test_start_time))
        
        # Check if test actually passed by examining log
        if grep -q "TEST PASSED" "$test_log" || grep -q "UVM_INFO.*Test.*PASSED" "$test_log"; then
            log_success "Test $test_name PASSED (${test_duration}s)"
            echo "PASS" > "$RUN_DIR/${test_name}.result"
            return 0
        else
            log_error "Test $test_name FAILED (${test_duration}s)"
            echo "FAIL" > "$RUN_DIR/${test_name}.result"
            return 1
        fi
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "Test $test_name TIMEOUT after ${TIMEOUT}s"
            echo "TIMEOUT" > "$RUN_DIR/${test_name}.result"
        else
            log_error "Test $test_name FAILED with exit code $exit_code"
            echo "FAIL" > "$RUN_DIR/${test_name}.result"
        fi
        return 1
    fi
}

# Run tests in parallel
run_tests_parallel() {
    local tests=("$@")
    local pids=()
    local results=()
    
    log_info "Running ${#tests[@]} tests with $PARALLEL_JOBS parallel jobs"
    
    # Function to run test in background
    run_test_bg() {
        run_single_test "$1"
        echo $? > "$RUN_DIR/${1}.exit_code"
    }
    
    # Start tests in parallel batches
    local test_index=0
    while [[ $test_index -lt ${#tests[@]} ]]; do
        # Start batch of parallel jobs
        local batch_pids=()
        for ((i=0; i<PARALLEL_JOBS && test_index<${#tests[@]}; i++)); do
            local test_name="${tests[$test_index]}"
            run_test_bg "$test_name" &
            batch_pids+=($!)
            ((test_index++))
        done
        
        # Wait for batch to complete
        for pid in "${batch_pids[@]}"; do
            wait $pid
        done
    done
}

# Run tests sequentially
run_tests_sequential() {
    local tests=("$@")
    local passed=0
    local failed=0
    
    log_info "Running ${#tests[@]} tests sequentially"
    
    for test_name in "${tests[@]}"; do
        if run_single_test "$test_name"; then
            ((passed++))
        else
            ((failed++))
        fi
    done
    
    log_info "Sequential run complete: $passed passed, $failed failed"
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    local report_file="$RUN_DIR/test_report.html"
    local summary_file="$RUN_DIR/test_summary.txt"
    
    # Count results
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local timeout_tests=0
    
    for result_file in "$RUN_DIR"/*.result; do
        if [[ -f "$result_file" ]]; then
            ((total_tests++))
            case $(cat "$result_file") in
                PASS) ((passed_tests++)) ;;
                FAIL) ((failed_tests++)) ;;
                TIMEOUT) ((timeout_tests++)) ;;
            esac
        fi
    done
    
    # Generate summary
    cat > "$summary_file" << EOF
RISC-V AI Accelerator UVM Test Report
=====================================

Run Information:
- Timestamp: $TIMESTAMP
- Simulator: $SIMULATOR
- Verbosity: $VERBOSITY
- Waves: $WAVES
- Coverage: $COVERAGE
- Parallel Jobs: $PARALLEL_JOBS

Test Results:
- Total Tests: $total_tests
- Passed: $passed_tests
- Failed: $failed_tests
- Timeout: $timeout_tests
- Pass Rate: $(( passed_tests * 100 / total_tests ))%

Test Details:
EOF
    
    # Add individual test results
    for result_file in "$RUN_DIR"/*.result; do
        if [[ -f "$result_file" ]]; then
            local test_name=$(basename "$result_file" .result)
            local result=$(cat "$result_file")
            echo "- $test_name: $result" >> "$summary_file"
        fi
    done
    
    # Generate HTML report
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>RISC-V AI Accelerator Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .timeout { color: orange; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RISC-V AI Accelerator UVM Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Simulator: $SIMULATOR | Verbosity: $VERBOSITY | Waves: $WAVES | Coverage: $COVERAGE</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: $total_tests</p>
        <p class="pass">Passed: $passed_tests</p>
        <p class="fail">Failed: $failed_tests</p>
        <p class="timeout">Timeout: $timeout_tests</p>
        <p>Pass Rate: $(( passed_tests * 100 / total_tests ))%</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr><th>Test Name</th><th>Result</th><th>Log File</th></tr>
EOF
    
    for result_file in "$RUN_DIR"/*.result; do
        if [[ -f "$result_file" ]]; then
            local test_name=$(basename "$result_file" .result)
            local result=$(cat "$result_file")
            local log_file="${test_name}.log"
            local css_class=""
            
            case $result in
                PASS) css_class="pass" ;;
                FAIL) css_class="fail" ;;
                TIMEOUT) css_class="timeout" ;;
            esac
            
            echo "<tr><td>$test_name</td><td class=\"$css_class\">$result</td><td><a href=\"$log_file\">$log_file</a></td></tr>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF
    </table>
</body>
</html>
EOF
    
    log_success "Test report generated: $report_file"
    log_success "Test summary: $summary_file"
    
    # Display summary
    cat "$summary_file"
}

# Generate coverage report
generate_coverage_report() {
    if [[ "$COVERAGE" == "1" ]]; then
        log_info "Generating coverage report..."
        cd "$SCRIPT_DIR"
        if make coverage > "$RUN_DIR/coverage.log" 2>&1; then
            log_success "Coverage report generated"
        else
            log_warn "Coverage report generation failed. Check $RUN_DIR/coverage.log"
        fi
    fi
}

# Main function
main() {
    # Default values
    TEST_CATEGORY="all"
    SPECIFIC_TEST=""
    
    # Parse arguments
    parse_args "$@"
    
    # Setup
    setup_directories
    check_prerequisites
    
    # Determine which tests to run
    local tests_to_run=()
    
    if [[ -n "$SPECIFIC_TEST" ]]; then
        tests_to_run=("$SPECIFIC_TEST")
    else
        case $TEST_CATEGORY in
            smoke)
                tests_to_run=("${SMOKE_TESTS[@]}")
                ;;
            basic)
                tests_to_run=("${BASIC_TESTS[@]}")
                ;;
            advanced)
                tests_to_run=("${ADVANCED_TESTS[@]}")
                ;;
            all|*)
                tests_to_run=("${ALL_TESTS[@]}")
                ;;
        esac
    fi
    
    log_info "Selected tests: ${tests_to_run[*]}"
    
    # Compile design
    compile_design
    
    # Run tests
    local start_time=$(date +%s)
    
    if [[ ${#tests_to_run[@]} -gt 1 && $PARALLEL_JOBS -gt 1 ]]; then
        run_tests_parallel "${tests_to_run[@]}"
    else
        run_tests_sequential "${tests_to_run[@]}"
    fi
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    log_info "Total execution time: ${total_duration}s"
    
    # Generate reports
    generate_report
    generate_coverage_report
    
    # Final status
    local failed_count=$(find "$RUN_DIR" -name "*.result" -exec grep -l "FAIL\|TIMEOUT" {} \; | wc -l)
    
    if [[ $failed_count -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "$failed_count test(s) failed"
        exit 1
    fi
}

# Run main function
main "$@"