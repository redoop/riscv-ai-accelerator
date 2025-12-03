# Flash/PSRAM 扩展实施 Checklist

## 项目信息
- **目标**: 添加 SPI Flash (16MB) 和 PSRAM (8MB) 支持
- **预计工期**: 8-9 天
- **复杂度**: 🟡 中等
- **状态**: 📋 待开始

---

## Phase 1: SPI Flash 控制器 (3天)

### Day 1: 控制器开发 ✅

#### 1.1 创建 SPIFlash 模块
- [ ] 创建文件 `chisel/src/main/scala/peripherals/SPIFlash.scala`
- [ ] 定义 IO 接口
  - [ ] 寄存器接口 (SimpleRegIO)
  - [ ] SPI 物理接口 (CLK, MOSI, MISO, CS)
- [ ] 实现寄存器
  - [ ] 0x00: CMD - 命令寄存器
  - [ ] 0x04: ADDR - 地址寄存器 (24-bit)
  - [ ] 0x08: DATA - 数据寄存器
  - [ ] 0x0C: CTRL - 控制寄存器 (start, busy, done)
  - [ ] 0x10: STATUS - 状态寄存器

#### 1.2 实现 SPI 协议
- [ ] SPI 时钟生成 (100MHz → 25MHz)
- [ ] 状态机设计
  - [ ] IDLE: 空闲状态
  - [ ] COMMAND: 发送命令字节
  - [ ] ADDRESS: 发送 24-bit 地址
  - [ ] DATA: 读取/写入数据
  - [ ] DONE: 完成状态
- [ ] 支持的命令
  - [ ] 0x03: READ (标准读取)
  - [ ] 0x0B: FAST_READ (快速读取)
  - [ ] 0x02: PAGE_PROGRAM (页编程)
  - [ ] 0x20: SECTOR_ERASE (扇区擦除)

#### 1.3 代码审查
- [ ] 代码风格检查
- [ ] 逻辑正确性验证
- [ ] 时序分析
- [ ] 提交代码到 git

**预期输出**: `SPIFlash.scala` (~300 行)

---

### Day 2: 集成和测试 ✅

#### 2.1 SoC 集成
- [ ] 修改 `EdgeAiSoCSimple.scala`
  - [ ] 添加 Flash 模块实例
  - [ ] 连接 IO 端口
  - [ ] 添加地址解码 (0x30000000-0x30FFFFFF)
- [ ] 更新内存映射文档
- [ ] 验证引脚分配

#### 2.2 创建测试用例
- [ ] 创建 `chisel/src/test/scala/SPIFlashTest.scala`
- [ ] 测试用例
  - [ ] 基本读取测试
  - [ ] 快速读取测试
  - [ ] 页编程测试
  - [ ] 扇区擦除测试
  - [ ] 边界条件测试
- [ ] 运行测试
  ```bash
  cd chisel
  sbt "testOnly riscv.ai.peripherals.SPIFlashTest"
  ```

#### 2.3 波形验证
- [ ] 生成 VCD 波形文件
- [ ] 验证 SPI 时序
  - [ ] 时钟频率: 25 MHz
  - [ ] 时钟相位: CPOL=0, CPHA=0
  - [ ] CS 信号正确
- [ ] 验证命令序列
- [ ] 验证数据传输

**预期输出**: 
- 修改的 `EdgeAiSoCSimple.scala` (+20 行)
- `SPIFlashTest.scala` (~200 行)
- 测试报告

---

### Day 3: 软件驱动 ✅

#### 3.1 HAL 层扩展
- [ ] 修改 `chisel/software/lib/hal.h`
  - [ ] 添加 Flash 寄存器定义
  - [ ] 添加 Flash 函数声明
- [ ] 修改 `chisel/software/lib/hal.c`
  - [ ] `flash_init()` - 初始化
  - [ ] `flash_read()` - 读取单字
  - [ ] `flash_read_block()` - 块读取
  - [ ] `flash_write_page()` - 页编程
  - [ ] `flash_erase_sector()` - 扇区擦除

#### 3.2 创建测试程序
- [ ] 创建 `chisel/software/examples/flash_test.c`
- [ ] 测试功能
  - [ ] Flash 初始化
  - [ ] 读取测试
  - [ ] 写入测试
  - [ ] 擦除测试
  - [ ] 性能测试
- [ ] 编译测试
  ```bash
  cd chisel/software
  make flash_test
  ```

#### 3.3 文档更新
- [ ] 更新 `chisel/software/README.md`
- [ ] 添加 Flash API 文档
- [ ] 添加使用示例
- [ ] 更新内存映射图

**预期输出**:
- 修改的 `hal.h` (+20 行)
- 修改的 `hal.c` (+100 行)
- `flash_test.c` (~150 行)

---

## Phase 2: PSRAM 控制器 (4天)

### Day 4: 控制器开发 (基础) ✅

#### 4.1 创建 PSRAM 模块
- [ ] 创建文件 `chisel/src/main/scala/peripherals/PSRAM.scala`
- [ ] 定义 IO 接口
  - [ ] 寄存器接口 (SimpleRegIO)
  - [ ] SPI/QPI 物理接口 (CLK, CS, SIO[3:0])
- [ ] 实现寄存器
  - [ ] 0x00: CMD - 命令寄存器
  - [ ] 0x04: ADDR - 地址寄存器 (24-bit)
  - [ ] 0x08: DATA - 数据寄存器
  - [ ] 0x0C: CTRL - 控制寄存器
  - [ ] 0x10: STATUS - 状态寄存器
  - [ ] 0x14: CONFIG - 配置寄存器 (SPI/QPI 模式)

#### 4.2 实现标准 SPI 模式
- [ ] SPI 时钟生成 (100MHz → 50MHz)
- [ ] 状态机设计
  - [ ] IDLE: 空闲
  - [ ] COMMAND: 发送命令
  - [ ] ADDRESS: 发送地址
  - [ ] WAIT: Dummy cycles
  - [ ] DATA: 数据传输
  - [ ] DONE: 完成
- [ ] 支持的命令
  - [ ] 0x03: READ
  - [ ] 0x0B: FAST_READ
  - [ ] 0x02: WRITE

#### 4.3 代码审查
- [ ] 代码风格检查
- [ ] 逻辑验证
- [ ] 提交代码

**预期输出**: `PSRAM.scala` (~250 行, SPI 模式)

---

### Day 5: Quad SPI 支持 ✅

#### 5.1 实现 Quad SPI 模式
- [ ] 添加 QPI 状态机
- [ ] 实现 4-bit 并行传输
- [ ] 支持的命令
  - [ ] 0xEB: QUAD_READ
  - [ ] 0x38: QUAD_WRITE
  - [ ] 0x35: ENTER_QPI
  - [ ] 0xF5: EXIT_QPI

#### 5.2 模式切换
- [ ] SPI → QPI 切换逻辑
- [ ] QPI → SPI 切换逻辑
- [ ] 模式状态保持
- [ ] 错误处理

#### 5.3 性能优化
- [ ] 流水线优化
- [ ] 缓冲优化
- [ ] 时序优化

**预期输出**: `PSRAM.scala` (~400 行, 完整版)

---

### Day 6: 集成和测试 ✅

#### 6.1 SoC 集成
- [ ] 修改 `EdgeAiSoCSimple.scala`
  - [ ] 添加 PSRAM 模块实例
  - [ ] 连接 IO 端口 (包括 Analog 类型)
  - [ ] 添加地址解码 (0x04000000-0x047FFFFF)
- [ ] 更新内存映射
- [ ] 验证引脚分配

#### 6.2 创建测试用例
- [ ] 创建 `chisel/src/test/scala/PSRAMTest.scala`
- [ ] SPI 模式测试
  - [ ] 读取测试
  - [ ] 写入测试
  - [ ] 读写一致性
- [ ] QPI 模式测试
  - [ ] 模式切换
  - [ ] Quad 读取
  - [ ] Quad 写入
  - [ ] 性能对比
- [ ] 运行测试
  ```bash
  sbt "testOnly riscv.ai.peripherals.PSRAMTest"
  ```

#### 6.3 波形验证
- [ ] 生成 VCD 波形
- [ ] 验证 SPI 时序 (50 MHz)
- [ ] 验证 QPI 时序
- [ ] 验证数据完整性

**预期输出**:
- 修改的 `EdgeAiSoCSimple.scala` (+30 行)
- `PSRAMTest.scala` (~250 行)

---

### Day 7: 软件驱动和验证 ✅

#### 7.1 HAL 层扩展
- [ ] 修改 `hal.h`
  - [ ] PSRAM 寄存器定义
  - [ ] PSRAM 函数声明
- [ ] 修改 `hal.c`
  - [ ] `psram_init()` - 初始化
  - [ ] `psram_read()` - 读取
  - [ ] `psram_write()` - 写入
  - [ ] `psram_read_block()` - 块读取
  - [ ] `psram_write_block()` - 块写入
  - [ ] `psram_enable_qpi()` - 启用 QPI
  - [ ] `psram_disable_qpi()` - 禁用 QPI

#### 7.2 创建测试程序
- [ ] 创建 `chisel/software/examples/psram_test.c`
- [ ] 测试功能
  - [ ] PSRAM 初始化
  - [ ] SPI 模式读写
  - [ ] QPI 模式读写
  - [ ] 性能对比
  - [ ] 内存测试 (walking 1s/0s)
- [ ] 编译测试
  ```bash
  make psram_test
  ```

#### 7.3 集成测试
- [ ] 创建 `chisel/software/examples/storage_demo.c`
- [ ] 综合测试
  - [ ] Flash + PSRAM 协同工作
  - [ ] 从 Flash 加载到 PSRAM
  - [ ] 大数据处理演示
  - [ ] 性能基准测试

**预期输出**:
- 修改的 `hal.h` (+15 行)
- 修改的 `hal.c` (+130 行)
- `psram_test.c` (~200 行)
- `storage_demo.c` (~150 行)

---

## Phase 3: 文档和优化 (1天)

### Day 8: 文档更新 ✅

#### 8.1 更新主文档
- [ ] 更新 `README.md`
  - [ ] 添加 Flash/PSRAM 特性说明
  - [ ] 更新内存映射表
  - [ ] 更新性能指标
- [ ] 更新 `chisel/README.md`
  - [ ] 添加外设说明
  - [ ] 更新架构图
- [ ] 更新 `chisel/software/README.md`
  - [ ] 添加 API 文档
  - [ ] 添加使用示例

#### 8.2 创建专项文档
- [ ] 创建 `FLASH_PSRAM_GUIDE.md`
  - [ ] 硬件接口说明
  - [ ] 寄存器映射
  - [ ] 软件 API
  - [ ] 使用示例
  - [ ] 性能优化建议
  - [ ] 故障排除

#### 8.3 更新测试文档
- [ ] 更新 `TESTING.md`
  - [ ] 添加 Flash 测试说明
  - [ ] 添加 PSRAM 测试说明
- [ ] 创建测试报告
  - [ ] `FLASH_TEST_REPORT.md`
  - [ ] `PSRAM_TEST_REPORT.md`

**预期输出**:
- 更新的 README 文件 (3个)
- `FLASH_PSRAM_GUIDE.md` (~500 行)
- 测试报告 (2个)

---

### Day 8-9: 性能优化和验证 ✅

#### 9.1 性能优化
- [ ] Flash 读取优化
  - [ ] 实现预取缓存
  - [ ] 优化时钟频率
  - [ ] 减少延迟
- [ ] PSRAM 优化
  - [ ] QPI 模式默认启用
  - [ ] 优化突发传输
  - [ ] 缓存策略

#### 9.2 综合验证
- [ ] 运行完整测试套件
  ```bash
  cd chisel
  sbt test
  cd software
  make all
  ./test_all.sh
  ```
- [ ] 性能基准测试
  - [ ] Flash 读取速度
  - [ ] PSRAM 读写速度
  - [ ] 延迟测量
- [ ] 资源使用分析
  - [ ] 面积增加
  - [ ] 功耗增加
  - [ ] 时序影响

#### 9.3 代码审查和清理
- [ ] 代码风格统一
- [ ] 注释完善
- [ ] 移除调试代码
- [ ] 优化代码结构

**预期输出**:
- 性能测试报告
- 资源使用报告
- 优化后的代码

---

## 验收标准

### 功能验收 ✅
- [ ] SPI Flash 控制器
  - [ ] 读取功能正常
  - [ ] 写入功能正常
  - [ ] 擦除功能正常
  - [ ] 时序符合规范
- [ ] PSRAM 控制器
  - [ ] SPI 模式正常
  - [ ] QPI 模式正常
  - [ ] 读写一致性
  - [ ] 性能达标

### 测试验收 ✅
- [ ] 单元测试通过率 100%
- [ ] 集成测试通过
- [ ] 软件测试通过
- [ ] 性能测试达标

### 文档验收 ✅
- [ ] API 文档完整
- [ ] 使用指南清晰
- [ ] 测试报告详细
- [ ] 示例代码可运行

### 性能验收 ✅
- [ ] Flash 读取: ≥ 3 MB/s
- [ ] PSRAM (SPI): ≥ 6 MB/s
- [ ] PSRAM (QPI): ≥ 20 MB/s
- [ ] 延迟: < 20 us

---

## 风险管理

### 技术风险
| 风险 | 概率 | 影响 | 缓解措施 | 负责人 |
|------|------|------|----------|--------|
| SPI 时序问题 | 中 | 中 | 参考 LCD SPI 实现 | 开发者 |
| QPI 实现复杂 | 中 | 中 | 先完成 SPI，再扩展 | 开发者 |
| 引脚冲突 | 低 | 高 | 提前规划引脚分配 | 架构师 |
| 测试不充分 | 中 | 高 | 完整测试计划 | 测试工程师 |

### 进度风险
| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 开发延期 | 中 | 中 | 预留缓冲时间 |
| 测试延期 | 低 | 中 | 并行测试开发 |
| 文档延期 | 低 | 低 | 边开发边文档 |

---

## 资源需求

### 人力资源
- **开发工程师**: 1 人 × 8-9 天
- **测试工程师**: 0.5 人 × 2 天 (并行)
- **文档工程师**: 0.5 人 × 1 天 (并行)

### 硬件资源
- **开发板**: 支持 SPI 接口
- **Flash 芯片**: 16 MB SPI Flash (可选)
- **PSRAM 芯片**: 8 MB PSRAM (可选)
- **逻辑分析仪**: 验证 SPI 时序 (可选)

### 软件工具
- **Chisel/Scala**: RTL 开发
- **SBT**: 构建和测试
- **RISC-V GCC**: 软件编译
- **GTKWave**: 波形查看
- **Git**: 版本控制

---

## 交付清单

### 代码交付
- [ ] `peripherals/SPIFlash.scala` (~300 行)
- [ ] `peripherals/PSRAM.scala` (~400 行)
- [ ] 修改的 `EdgeAiSoCSimple.scala` (+50 行)
- [ ] `SPIFlashTest.scala` (~200 行)
- [ ] `PSRAMTest.scala` (~250 行)
- [ ] 修改的 `hal.h` (+35 行)
- [ ] 修改的 `hal.c` (+230 行)
- [ ] `flash_test.c` (~150 行)
- [ ] `psram_test.c` (~200 行)
- [ ] `storage_demo.c` (~150 行)

### 文档交付
- [ ] 更新的 README 文件 (3个)
- [ ] `FLASH_PSRAM_GUIDE.md`
- [ ] `FLASH_TEST_REPORT.md`
- [ ] `PSRAM_TEST_REPORT.md`
- [ ] API 文档

### 测试交付
- [ ] 单元测试用例
- [ ] 集成测试用例
- [ ] 性能测试报告
- [ ] 波形文件 (VCD)

---

## 后续计划

### v0.5 集成
- [ ] 合并到主分支
- [ ] 创建 v0.5 标签
- [ ] 发布 Release Notes

### v0.6 优化
- [ ] DMA 支持
- [ ] 缓存优化
- [ ] 性能调优

### v1.0 完善
- [ ] FPGA 验证
- [ ] ASIC 综合
- [ ] 应用示例

---

## 签署确认

| 角色 | 姓名 | 签名 | 日期 |
|------|------|------|------|
| 项目经理 | | | |
| 开发负责人 | | | |
| 测试负责人 | | | |
| 文档负责人 | | | |

---

**创建日期**: 2025年12月3日  
**最后更新**: 2025年12月3日  
**版本**: v1.0  
**状态**: 📋 待执行
