#!/bin/bash

set -e

echo "=== Flash/PSRAM 扩展项目启动 ==="
echo ""
echo "项目信息:"
echo "  - 目标: 添加 SPI Flash (16MB) 和 PSRAM (8MB)"
echo "  - 预计工期: 8-9 天"
echo "  - 复杂度: 🟡 中等"
echo ""

# 创建工作目录
echo "创建工作目录..."
mkdir -p chisel/src/main/scala/peripherals
mkdir -p chisel/src/test/scala/peripherals
mkdir -p chisel/software/examples
mkdir -p docs/flash_psram

echo "✓ 目录创建完成"
echo ""

# 显示 Phase 1 任务
echo "=== Phase 1: SPI Flash 控制器 (3天) ==="
echo ""
echo "Day 1: 控制器开发"
echo "  [ ] 创建 SPIFlash.scala"
echo "  [ ] 实现 SPI 协议"
echo "  [ ] 支持 READ/WRITE/ERASE 命令"
echo ""
echo "Day 2: 集成和测试"
echo "  [ ] SoC 集成"
echo "  [ ] 创建测试用例"
echo "  [ ] 波形验证"
echo ""
echo "Day 3: 软件驱动"
echo "  [ ] HAL 层扩展"
echo "  [ ] 创建测试程序"
echo "  [ ] 文档更新"
echo ""

# 创建进度跟踪文件
cat > FLASH_PSRAM_PROGRESS.md << 'EOF'
# Flash/PSRAM 扩展进度跟踪

**开始日期**: $(date +%Y-%m-%d)
**状态**: 🔄 进行中

## 当前进度

### Phase 1: SPI Flash 控制器 (0/3 天)
- [ ] Day 1: 控制器开发
- [ ] Day 2: 集成和测试
- [ ] Day 3: 软件驱动

### Phase 2: PSRAM 控制器 (0/4 天)
- [ ] Day 4: 控制器开发 (基础)
- [ ] Day 5: Quad SPI 支持
- [ ] Day 6: 集成和测试
- [ ] Day 7: 软件驱动和验证

### Phase 3: 文档和优化 (0/1 天)
- [ ] Day 8-9: 文档更新和性能优化

## 每日更新

### $(date +%Y-%m-%d)
- 项目启动
- 创建工作目录
- 准备开发环境

EOF

echo "✓ 进度跟踪文件创建: FLASH_PSRAM_PROGRESS.md"
echo ""

echo "=== 下一步行动 ==="
echo ""
echo "1. 开始 Day 1 开发:"
echo "   cd chisel/src/main/scala/peripherals"
echo "   # 创建 SPIFlash.scala"
echo ""
echo "2. 参考现有代码:"
echo "   cat chisel/src/main/scala/peripherals/TFTLCD.scala"
echo ""
echo "3. 查看详细计划:"
echo "   cat FLASH_PSRAM_CHECKLIST.md"
echo ""
echo "4. 跟踪进度:"
echo "   cat FLASH_PSRAM_PROGRESS.md"
echo ""

echo "✅ 项目启动完成！"
