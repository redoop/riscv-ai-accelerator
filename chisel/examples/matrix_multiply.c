/**
 * RISC-V AI 加速器示例程序
 * 演示如何使用 AI 加速器进行矩阵乘法
 */

#include <stdint.h>

// AI 加速器基地址
#define AI_BASE_ADDR    0x80000000

// 寄存器偏移
#define MATRIX_A_OFFSET 0x000
#define MATRIX_B_OFFSET 0x100
#define RESULT_OFFSET   0x200
#define CTRL_REG_OFFSET 0x300
#define STATUS_REG_OFFSET 0x304

// 矩阵大小
#define MATRIX_SIZE 8

// 寄存器访问宏
#define AI_REG(offset) (*(volatile uint32_t*)(AI_BASE_ADDR + (offset)))
#define MATRIX_A(row, col) AI_REG(MATRIX_A_OFFSET + ((row) * MATRIX_SIZE + (col)) * 4)
#define MATRIX_B(row, col) AI_REG(MATRIX_B_OFFSET + ((row) * MATRIX_SIZE + (col)) * 4)
#define RESULT(row, col) AI_REG(RESULT_OFFSET + ((row) * MATRIX_SIZE + (col)) * 4)
#define CTRL_REG AI_REG(CTRL_REG_OFFSET)
#define STATUS_REG AI_REG(STATUS_REG_OFFSET)

// 控制寄存器位定义
#define CTRL_START_BIT  (1 << 0)

// 状态寄存器位定义
#define STATUS_BUSY_BIT (1 << 0)
#define STATUS_DONE_BIT (1 << 1)

/**
 * 初始化矩阵 A
 */
void init_matrix_a(void) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // 简单的测试模式: A[i][j] = i + j
            MATRIX_A(i, j) = i + j;
        }
    }
}

/**
 * 初始化矩阵 B
 */
void init_matrix_b(void) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // 简单的测试模式: B[i][j] = i * j
            MATRIX_B(i, j) = i * j;
        }
    }
}

/**
 * 启动矩阵乘法
 */
void start_matrix_multiply(void) {
    CTRL_REG = CTRL_START_BIT;
}

/**
 * 等待矩阵乘法完成
 */
void wait_for_completion(void) {
    while ((STATUS_REG & STATUS_DONE_BIT) == 0) {
        // 忙等待
        // 在实际应用中，可以使用中断或轮询其他任务
    }
}

/**
 * 读取结果矩阵
 */
void read_result_matrix(int32_t result[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i][j] = RESULT(i, j);
        }
    }
}

/**
 * 软件实现的矩阵乘法 (用于验证)
 */
void software_matrix_multiply(
    const int32_t a[MATRIX_SIZE][MATRIX_SIZE],
    const int32_t b[MATRIX_SIZE][MATRIX_SIZE],
    int32_t result[MATRIX_SIZE][MATRIX_SIZE]
) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

/**
 * 打印矩阵 (简化版本，实际需要 printf 支持)
 */
void print_matrix(const char* name, const int32_t matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    // 在实际硬件上，这需要 UART 或其他输出设备
    // 这里只是示意
    (void)name;
    (void)matrix;
}

/**
 * 验证结果
 */
int verify_result(
    const int32_t hw_result[MATRIX_SIZE][MATRIX_SIZE],
    const int32_t sw_result[MATRIX_SIZE][MATRIX_SIZE]
) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (hw_result[i][j] != sw_result[i][j]) {
                return 0; // 验证失败
            }
        }
    }
    return 1; // 验证成功
}

/**
 * 主函数
 */
int main(void) {
    int32_t hw_result[MATRIX_SIZE][MATRIX_SIZE];
    int32_t sw_result[MATRIX_SIZE][MATRIX_SIZE];
    int32_t matrix_a[MATRIX_SIZE][MATRIX_SIZE];
    int32_t matrix_b[MATRIX_SIZE][MATRIX_SIZE];
    
    // 1. 初始化矩阵
    init_matrix_a();
    init_matrix_b();
    
    // 保存矩阵副本用于软件验证
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix_a[i][j] = MATRIX_A(i, j);
            matrix_b[i][j] = MATRIX_B(i, j);
        }
    }
    
    // 2. 启动硬件加速器
    start_matrix_multiply();
    
    // 3. 等待完成
    wait_for_completion();
    
    // 4. 读取硬件结果
    read_result_matrix(hw_result);
    
    // 5. 软件计算用于验证
    software_matrix_multiply(matrix_a, matrix_b, sw_result);
    
    // 6. 验证结果
    int success = verify_result(hw_result, sw_result);
    
    // 7. 返回结果
    return success ? 0 : 1;
}

/**
 * 性能测试函数
 */
void performance_test(void) {
    // 假设有计时器支持
    uint32_t start_time, end_time;
    
    // 初始化
    init_matrix_a();
    init_matrix_b();
    
    // 测量硬件加速器性能
    start_time = 0; // 读取计时器
    start_matrix_multiply();
    wait_for_completion();
    end_time = 0; // 读取计时器
    
    uint32_t hw_cycles = end_time - start_time;
    
    // 测量软件性能
    int32_t matrix_a[MATRIX_SIZE][MATRIX_SIZE];
    int32_t matrix_b[MATRIX_SIZE][MATRIX_SIZE];
    int32_t result[MATRIX_SIZE][MATRIX_SIZE];
    
    start_time = 0;
    software_matrix_multiply(matrix_a, matrix_b, result);
    end_time = 0;
    
    uint32_t sw_cycles = end_time - start_time;
    
    // 计算加速比
    uint32_t speedup = sw_cycles / hw_cycles;
    
    (void)speedup; // 在实际应用中输出结果
}

/**
 * 批量处理示例
 */
void batch_processing(void) {
    const int NUM_BATCHES = 10;
    
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        // 加载新的矩阵数据
        init_matrix_a();
        init_matrix_b();
        
        // 启动计算
        start_matrix_multiply();
        
        // 等待完成
        wait_for_completion();
        
        // 读取结果
        int32_t result[MATRIX_SIZE][MATRIX_SIZE];
        read_result_matrix(result);
        
        // 处理结果...
    }
}
