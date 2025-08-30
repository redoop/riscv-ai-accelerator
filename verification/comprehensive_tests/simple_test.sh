#!/bin/sh

# Simple Test Runner for RISC-V AI Accelerator
# Compatible with POSIX shell

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging functions
log_info() {
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

log_success() {
    printf "${GREEN}[PASS]${NC} %s\n" "$1"
}

log_error() {
    printf "${RED}[FAIL]${NC} %s\n" "$1"
}

log_warning() {
    printf "${YELLOW}[WARN]${NC} %s\n" "$1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v verilator >/dev/null 2>&1; then
        log_error "Verilator not found. Please install Verilator."
        return 1
    fi
    
    if ! command -v make >/dev/null 2>&1; then
        log_error "Make not found. Please install make."
        return 1
    fi
    
    log_success "Prerequisites check passed"
    return 0
}

# Run syntax check
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

# Run a single test
run_test() {
    test_name="$1"
    log_info "Running $test_name test..."
    
    cd "$SCRIPT_DIR"
    if make "$test_name" > "$LOG_DIR/${test_name}_test.log" 2>&1; then
        log_success "$test_name test passed"
        return 0
    else
        log_error "$test_name test failed (see $LOG_DIR/${test_name}_test.log)"
        return 1
    fi
}

# Main function
main() {
    log_info "Starting RISC-V AI Accelerator test suite"
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Parse arguments
    case "${1:-all}" in
        "syntax")
            run_syntax_check
            exit $?
            ;;
        "comprehensive")
            run_syntax_check || log_warning "Syntax check failed, continuing..."
            run_test "comprehensive"
            exit $?
            ;;
        "ai_instructions")
            run_syntax_check || log_warning "Syntax check failed, continuing..."
            run_test "ai_instructions"
            exit $?
            ;;
        "performance")
            run_syntax_check || log_warning "Syntax check failed, continuing..."
            run_test "performance"
            exit $?
            ;;
        "integration")
            run_syntax_check || log_warning "Syntax check failed, continuing..."
            run_test "integration"
            exit $?
            ;;
        "all")
            log_info "Running all tests..."
            
            run_syntax_check || log_warning "Syntax check failed, continuing..."
            
            passed=0
            failed=0
            
            for test in comprehensive ai_instructions performance integration; do
                if run_test "$test"; then
                    passed=$((passed + 1))
                else
                    failed=$((failed + 1))
                fi
            done
            
            log_info "Test Summary: $passed passed, $failed failed"
            
            if [ $failed -eq 0 ]; then
                log_success "All tests passed!"
                exit 0
            else
                log_error "$failed tests failed"
                exit 1
            fi
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [TEST_NAME]"
            echo ""
            echo "Available tests:"
            echo "  syntax        - Run syntax check only"
            echo "  comprehensive - Run comprehensive functionality tests"
            echo "  ai_instructions - Run AI instruction tests"
            echo "  performance   - Run performance benchmarks"
            echo "  integration   - Run system integration tests"
            echo "  all           - Run all tests (default)"
            echo "  help          - Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown test: $1"
            echo "Use '$0 help' for available options"
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"