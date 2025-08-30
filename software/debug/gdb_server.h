/*
 * GDB Server for RISC-V AI Accelerator
 * Provides GDB remote debugging protocol support
 */

#ifndef GDB_SERVER_H
#define GDB_SERVER_H

#include "jtag_interface.h"
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// GDB server configuration
typedef struct {
    int port;                   // TCP port for GDB connection
    char* bind_address;         // IP address to bind to
    bool verbose;               // Enable verbose logging
    int max_connections;        // Maximum concurrent connections
    int timeout_ms;             // Connection timeout
} gdb_server_config_t;

// GDB client connection
typedef struct {
    int socket_fd;              // Client socket
    bool connected;             // Connection status
    char* buffer;               // Communication buffer
    size_t buffer_size;         // Buffer size
    pthread_t thread;           // Client thread
} gdb_client_t;

// GDB server instance
typedef struct {
    gdb_server_config_t config; // Server configuration
    jtag_interface_t* jtag;     // JTAG interface
    debug_target_t* target;     // Debug target
    
    int server_socket;          // Server socket
    bool running;               // Server running flag
    pthread_t server_thread;    // Server thread
    
    gdb_client_t* clients;      // Client connections
    int client_count;           // Number of active clients
    pthread_mutex_t clients_mutex; // Client list mutex
    
    // Target state
    bool target_halted;         // Target halt status
    uint64_t pc;                // Program counter
    uint64_t registers[32];     // General purpose registers
    
    // Breakpoints and watchpoints
    uint32_t* breakpoints;      // Breakpoint addresses
    int breakpoint_count;       // Number of breakpoints
    uint32_t* watchpoints;      // Watchpoint addresses
    int watchpoint_count;       // Number of watchpoints
} gdb_server_t;

// GDB packet types
typedef enum {
    GDB_PACKET_ACK = '+',
    GDB_PACKET_NACK = '-',
    GDB_PACKET_INTERRUPT = 0x03,
    GDB_PACKET_COMMAND = '$'
} gdb_packet_type_t;

// Function prototypes

// Server management
int gdb_server_init(gdb_server_t* server, const gdb_server_config_t* config,
                   jtag_interface_t* jtag, debug_target_t* target);
int gdb_server_start(gdb_server_t* server);
int gdb_server_stop(gdb_server_t* server);
void gdb_server_cleanup(gdb_server_t* server);

// Client management
int gdb_server_accept_client(gdb_server_t* server);
void gdb_server_disconnect_client(gdb_server_t* server, gdb_client_t* client);
void gdb_server_broadcast_stop(gdb_server_t* server);

// Packet handling
int gdb_packet_receive(gdb_client_t* client, char* packet, size_t max_size);
int gdb_packet_send(gdb_client_t* client, const char* packet);
int gdb_packet_send_ok(gdb_client_t* client);
int gdb_packet_send_error(gdb_client_t* client, int error_code);

// Command handlers
int gdb_handle_query(gdb_server_t* server, gdb_client_t* client, const char* query);
int gdb_handle_read_registers(gdb_server_t* server, gdb_client_t* client);
int gdb_handle_write_registers(gdb_server_t* server, gdb_client_t* client, const char* data);
int gdb_handle_read_memory(gdb_server_t* server, gdb_client_t* client, 
                          uint64_t address, size_t length);
int gdb_handle_write_memory(gdb_server_t* server, gdb_client_t* client,
                           uint64_t address, const uint8_t* data, size_t length);
int gdb_handle_continue(gdb_server_t* server, gdb_client_t* client, uint64_t address);
int gdb_handle_step(gdb_server_t* server, gdb_client_t* client, uint64_t address);
int gdb_handle_breakpoint(gdb_server_t* server, gdb_client_t* client,
                         char type, uint64_t address, size_t length);
int gdb_handle_detach(gdb_server_t* server, gdb_client_t* client);

// Target state management
int gdb_update_target_state(gdb_server_t* server);
int gdb_halt_target(gdb_server_t* server);
int gdb_resume_target(gdb_server_t* server, uint64_t address);
int gdb_step_target(gdb_server_t* server, uint64_t address);

// Utility functions
uint8_t gdb_checksum(const char* data);
int gdb_hex_to_bin(const char* hex, uint8_t* bin, size_t max_len);
int gdb_bin_to_hex(const uint8_t* bin, size_t len, char* hex, size_t max_hex_len);
uint64_t gdb_parse_hex_uint64(const char* hex);

// Default configuration
gdb_server_config_t gdb_server_default_config(void);

// Error codes
#define GDB_SUCCESS             0
#define GDB_ERROR_INIT          -1
#define GDB_ERROR_SOCKET        -2
#define GDB_ERROR_BIND          -3
#define GDB_ERROR_LISTEN        -4
#define GDB_ERROR_ACCEPT        -5
#define GDB_ERROR_PROTOCOL      -6
#define GDB_ERROR_TARGET        -7
#define GDB_ERROR_MEMORY        -8
#define GDB_ERROR_TIMEOUT       -9

#ifdef __cplusplus
}
#endif

#endif // GDB_SERVER_H