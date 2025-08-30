# Makefile for RISC-V AI Accelerator Debug Tools

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -std=c99 -Wall -Wextra -O2 -g
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -g
LDFLAGS = -lpthread -lm

# Include directories
INCLUDES = -I. -I../lib -I../compiler

# Library directories and libraries
LIBDIRS = -L../lib
LIBS = -ltpu

# Source files
JTAG_SOURCES = jtag_interface.c
GDB_SOURCES = gdb_server.c
PERF_SOURCES = hardware_counters.c performance_analyzer.c
TEST_SOURCES = test_debug_tools.c

# Object files
JTAG_OBJECTS = $(JTAG_SOURCES:.c=.o)
GDB_OBJECTS = $(GDB_SOURCES:.c=.o)
PERF_OBJECTS = $(PERF_SOURCES:.c=.o)
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

# Target executables
TARGETS = libdebug.a gdb_server test_debug_tools perf_analyzer

# Default target
all: $(TARGETS)

# Static library containing all debug functionality
libdebug.a: $(JTAG_OBJECTS) $(GDB_OBJECTS) $(PERF_OBJECTS)
	ar rcs $@ $^
	ranlib $@

# GDB server executable
gdb_server: gdb_server_main.o libdebug.a
	$(CC) $(CFLAGS) -o $@ $^ $(LIBDIRS) $(LIBS) $(LDFLAGS)

# Performance analyzer tool
perf_analyzer: perf_analyzer_main.o libdebug.a
	$(CC) $(CFLAGS) -o $@ $^ $(LIBDIRS) $(LIBS) $(LDFLAGS)

# Test executable
test_debug_tools: $(TEST_OBJECTS) libdebug.a
	$(CC) $(CFLAGS) -o $@ $^ $(LIBDIRS) $(LIBS) $(LDFLAGS)

# Object file compilation
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create main executables if they don't exist
gdb_server_main.c:
	@echo "Creating GDB server main..."
	@echo '#include "gdb_server.h"' > $@
	@echo '#include <stdio.h>' >> $@
	@echo '#include <stdlib.h>' >> $@
	@echo '#include <signal.h>' >> $@
	@echo '' >> $@
	@echo 'static gdb_server_t server;' >> $@
	@echo 'static volatile int running = 1;' >> $@
	@echo '' >> $@
	@echo 'void signal_handler(int sig) {' >> $@
	@echo '    running = 0;' >> $@
	@echo '    gdb_server_stop(&server);' >> $@
	@echo '}' >> $@
	@echo '' >> $@
	@echo 'int main(int argc, char* argv[]) {' >> $@
	@echo '    gdb_server_config_t config = gdb_server_default_config();' >> $@
	@echo '    jtag_interface_t jtag;' >> $@
	@echo '    debug_target_t target;' >> $@
	@echo '' >> $@
	@echo '    if (argc > 1) config.port = atoi(argv[1]);' >> $@
	@echo '' >> $@
	@echo '    printf("Starting GDB server on port %d\\n", config.port);' >> $@
	@echo '' >> $@
	@echo '    if (jtag_init(&jtag, 1, 2, 3, 4, 5) != JTAG_SUCCESS) {' >> $@
	@echo '        fprintf(stderr, "Failed to initialize JTAG\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    if (debug_init(&jtag, &target) != JTAG_SUCCESS) {' >> $@
	@echo '        fprintf(stderr, "Failed to initialize debug target\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    if (gdb_server_init(&server, &config, &jtag, &target) != GDB_SUCCESS) {' >> $@
	@echo '        fprintf(stderr, "Failed to initialize GDB server\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    signal(SIGINT, signal_handler);' >> $@
	@echo '    signal(SIGTERM, signal_handler);' >> $@
	@echo '' >> $@
	@echo '    if (gdb_server_start(&server) != GDB_SUCCESS) {' >> $@
	@echo '        fprintf(stderr, "Failed to start GDB server\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    while (running) {' >> $@
	@echo '        sleep(1);' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    gdb_server_cleanup(&server);' >> $@
	@echo '    jtag_cleanup(&jtag);' >> $@
	@echo '    return 0;' >> $@
	@echo '}' >> $@

perf_analyzer_main.c:
	@echo "Creating performance analyzer main..."
	@echo '#include "performance_analyzer.h"' > $@
	@echo '#include <stdio.h>' >> $@
	@echo '#include <stdlib.h>' >> $@
	@echo '#include <string.h>' >> $@
	@echo '#include <unistd.h>' >> $@
	@echo '' >> $@
	@echo 'void print_usage(const char* prog_name) {' >> $@
	@echo '    printf("Usage: %s [options]\\n", prog_name);' >> $@
	@echo '    printf("Options:\\n");' >> $@
	@echo '    printf("  -s <session_name>  Start performance session\\n");' >> $@
	@echo '    printf("  -t <seconds>       Run for specified time\\n");' >> $@
	@echo '    printf("  -o <output_file>   Output file for report\\n");' >> $@
	@echo '    printf("  -c <counter_id>    Add performance counter\\n");' >> $@
	@echo '    printf("  -p                 Enable profiling\\n");' >> $@
	@echo '    printf("  -h                 Show this help\\n");' >> $@
	@echo '}' >> $@
	@echo '' >> $@
	@echo 'int main(int argc, char* argv[]) {' >> $@
	@echo '    perf_session_t session;' >> $@
	@echo '    profiling_data_t profiling;' >> $@
	@echo '    perf_report_t report;' >> $@
	@echo '    char session_name[256] = "default";' >> $@
	@echo '    char output_file[512] = "perf_report.txt";' >> $@
	@echo '    int run_time = 10;' >> $@
	@echo '    bool enable_profiling = false;' >> $@
	@echo '    int opt;' >> $@
	@echo '' >> $@
	@echo '    while ((opt = getopt(argc, argv, "s:t:o:c:ph")) != -1) {' >> $@
	@echo '        switch (opt) {' >> $@
	@echo '        case "s": strncpy(session_name, optarg, sizeof(session_name)-1); break;' >> $@
	@echo '        case "t": run_time = atoi(optarg); break;' >> $@
	@echo '        case "o": strncpy(output_file, optarg, sizeof(output_file)-1); break;' >> $@
	@echo '        case "c": /* Add counter - simplified */ break;' >> $@
	@echo '        case "p": enable_profiling = true; break;' >> $@
	@echo '        case "h": print_usage(argv[0]); return 0;' >> $@
	@echo '        default: print_usage(argv[0]); return 1;' >> $@
	@echo '        }' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    printf("Initializing performance monitoring...\\n");' >> $@
	@echo '    if (perf_init_counters() != PERF_SUCCESS) {' >> $@
	@echo '        fprintf(stderr, "Failed to initialize performance counters\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    printf("Creating session: %s\\n", session_name);' >> $@
	@echo '    if (perf_session_create(&session, session_name) != PERF_SUCCESS) {' >> $@
	@echo '        fprintf(stderr, "Failed to create performance session\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    perf_session_add_counter(&session, PERF_COUNTER_CYCLES);' >> $@
	@echo '    perf_session_add_counter(&session, PERF_COUNTER_INSTRUCTIONS);' >> $@
	@echo '    perf_session_add_counter(&session, PERF_COUNTER_CACHE_MISSES);' >> $@
	@echo '' >> $@
	@echo '    if (enable_profiling) {' >> $@
	@echo '        printf("Starting profiling...\\n");' >> $@
	@echo '        memset(&profiling, 0, sizeof(profiling));' >> $@
	@echo '        profiling.max_samples = 1000;' >> $@
	@echo '        profiling_start(&profiling, 10000);' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    printf("Starting performance session for %d seconds...\\n", run_time);' >> $@
	@echo '    perf_session_start(&session);' >> $@
	@echo '' >> $@
	@echo '    sleep(run_time);' >> $@
	@echo '' >> $@
	@echo '    printf("Stopping performance session...\\n");' >> $@
	@echo '    perf_session_stop(&session);' >> $@
	@echo '' >> $@
	@echo '    if (enable_profiling) {' >> $@
	@echo '        profiling_stop(&profiling);' >> $@
	@echo '        profiling_analyze(&profiling);' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    printf("Generating report...\\n");' >> $@
	@echo '    if (perf_analyze_session(&session, &report) == PERF_SUCCESS) {' >> $@
	@echo '        perf_generate_report(&report, output_file);' >> $@
	@echo '        printf("Report saved to: %s\\n", output_file);' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    perf_session_destroy(&session);' >> $@
	@echo '    if (enable_profiling) {' >> $@
	@echo '        profiling_cleanup(&profiling);' >> $@
	@echo '    }' >> $@
	@echo '' >> $@
	@echo '    return 0;' >> $@
	@echo '}' >> $@

# Phony targets
.PHONY: all clean install test benchmark docs

# Clean build artifacts
clean:
	rm -f *.o $(TARGETS) gdb_server_main.c perf_analyzer_main.c
	rm -f *.a *.so
	rm -f core dump.*

# Install targets
install: $(TARGETS)
	mkdir -p $(DESTDIR)/usr/local/bin
	mkdir -p $(DESTDIR)/usr/local/lib
	mkdir -p $(DESTDIR)/usr/local/include/riscv_ai_debug
	cp gdb_server perf_analyzer $(DESTDIR)/usr/local/bin/
	cp libdebug.a $(DESTDIR)/usr/local/lib/
	cp *.h $(DESTDIR)/usr/local/include/riscv_ai_debug/

# Run tests
test: test_debug_tools
	./test_debug_tools

# Run benchmarks
benchmark: perf_analyzer
	./perf_analyzer -s benchmark -t 5 -p -o benchmark_report.txt

# Generate documentation
docs:
	doxygen Doxyfile

# Create Doxyfile if it doesn't exist
Doxyfile:
	@echo "Creating Doxyfile..."
	@echo "PROJECT_NAME = \"RISC-V AI Debug Tools\"" > $@
	@echo "PROJECT_VERSION = 1.0" >> $@
	@echo "INPUT = ." >> $@
	@echo "OUTPUT_DIRECTORY = docs" >> $@
	@echo "GENERATE_HTML = YES" >> $@
	@echo "GENERATE_LATEX = NO" >> $@
	@echo "EXTRACT_ALL = YES" >> $@
	@echo "EXTRACT_PRIVATE = YES" >> $@
	@echo "EXTRACT_STATIC = YES" >> $@
	@echo "HAVE_DOT = YES" >> $@
	@echo "CALL_GRAPH = YES" >> $@
	@echo "CALLER_GRAPH = YES" >> $@

# Debug build
debug: CFLAGS += -DDEBUG -g3 -O0
debug: CXXFLAGS += -DDEBUG -g3 -O0
debug: $(TARGETS)

# Release build
release: CFLAGS += -DNDEBUG -O3
release: CXXFLAGS += -DNDEBUG -O3
release: $(TARGETS)

# Static analysis
analyze:
	cppcheck --enable=all --std=c99 *.c *.h
	clang-static-analyzer -analyze *.c

# Code formatting
format:
	clang-format -i *.c *.h

# Show build configuration
show-config:
	@echo "CC: $(CC)"
	@echo "CXX: $(CXX)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "LIBDIRS: $(LIBDIRS)"
	@echo "LIBS: $(LIBS)"
	@echo "LDFLAGS: $(LDFLAGS)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build all targets"
	@echo "  libdebug.a - Build debug library"
	@echo "  gdb_server - Build GDB server"
	@echo "  perf_analyzer - Build performance analyzer"
	@echo "  test_debug_tools - Build test suite"
	@echo "  clean      - Clean build artifacts"
	@echo "  install    - Install binaries and libraries"
	@echo "  test       - Run test suite"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  docs       - Generate documentation"
	@echo "  debug      - Build with debug symbols"
	@echo "  release    - Build optimized release"
	@echo "  analyze    - Run static analysis"
	@echo "  format     - Format source code"
	@echo "  show-config - Show build configuration"
	@echo "  help       - Show this help"