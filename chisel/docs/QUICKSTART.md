# Chisel 快速入门指南

## 🚀 5分钟上手Chisel

### 1. 环境准备
```bash
# macOS
brew install sbt scala

# Ubuntu/Debian
sudo apt update
sudo apt install sbt scala

# 验证安装
sbt --version
scala -version
```

### 2. 创建第一个Chisel模块
```scala
import chisel3._

// 简单的加法器
class Adder extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(8.W))
    val b = Input(UInt(8.W))
    val sum = Output(UInt(8.W))
  })
  
  io.sum := io.a + io.b
}
```

### 3. 编写测试
```scala
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AdderTest extends AnyFlatSpec with ChiselScalatestTester {
  "Adder" should "add two numbers" in {
    test(new Adder) { dut =>
      dut.io.a.poke(3.U)
      dut.io.b.poke(5.U)
      dut.io.sum.expect(8.U)
    }
  }
}
```

### 4. 运行测试
```bash
sbt test
```

## 📚 核心概念速览

### 数据类型
```scala
// 无符号整数
val unsigned = UInt(8.W)    // 8位无符号
val data = 42.U             // 字面量

// 有符号整数  
val signed = SInt(8.W)      // 8位有符号
val negative = (-5).S       // 负数字面量

// 布尔值
val flag = Bool()           // 布尔类型
val high = true.B           // 布尔字面量
```

### 寄存器和连线
```scala
// 寄存器 (时序逻辑)
val counter = RegInit(0.U(8.W))  // 初始值为0的8位寄存器
counter := counter + 1.U         // 下个时钟周期更新

// 连线 (组合逻辑)
val sum = Wire(UInt(8.W))        // 声明连线
sum := io.a + io.b               // 组合逻辑赋值
```

### 条件逻辑
```scala
// when语句 (类似if)
when(io.enable) {
  counter := counter + 1.U
}.elsewhen(io.reset) {
  counter := 0.U
}.otherwise {
  counter := counter
}

// Mux选择器
val result = Mux(io.select, io.a, io.b)  // select ? a : b
```

### Bundle接口
```scala
// 自定义接口
class MyInterface extends Bundle {
  val data = UInt(32.W)
  val valid = Bool()
  val ready = Bool()
}

// 使用接口
val io = IO(new Bundle {
  val input = Input(new MyInterface)
  val output = Output(new MyInterface)
})
```

## 🎯 常用模式

### 1. 状态机
```scala
val sIdle :: sWork :: sDone :: Nil = Enum(3)
val state = RegInit(sIdle)

switch(state) {
  is(sIdle) {
    when(io.start) { state := sWork }
  }
  is(sWork) {
    when(io.finished) { state := sDone }
  }
  is(sDone) {
    state := sIdle
  }
}
```

### 2. 存储器
```scala
val memory = Mem(1024, UInt(32.W))  // 1024个32位字

// 写入
when(io.writeEn) {
  memory(io.addr) := io.writeData
}

// 读取
io.readData := memory(io.addr)
```

### 3. 计数器
```scala
val counter = RegInit(0.U(8.W))

when(io.enable) {
  when(counter === 255.U) {
    counter := 0.U
  }.otherwise {
    counter := counter + 1.U
  }
}
```

## 🔧 生成Verilog

### 方法1: 使用ChiselStage
```scala
import chisel3.stage.ChiselStage

object Main extends App {
  (new ChiselStage).emitVerilog(new MyModule)
}
```

### 方法2: 使用sbt
```bash
sbt "runMain MyPackage.Main"
```

## 🧪 测试最佳实践

### 基本测试
```scala
test(new MyModule) { dut =>
  // 设置输入
  dut.io.input.poke(42.U)
  
  // 等待时钟周期
  dut.clock.step(1)
  
  // 检查输出
  dut.io.output.expect(42.U)
}
```

### 时序测试
```scala
test(new Counter) { dut =>
  dut.io.enable.poke(true.B)
  
  for (i <- 0 until 10) {
    dut.io.count.expect(i.U)
    dut.clock.step(1)
  }
}
```

### 波形生成
```scala
test(new MyModule).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
  // 测试代码...
  // 自动生成.vcd波形文件
}
```

## 📖 学习路径

### 第1周：基础语法
- [ ] Scala基础语法
- [ ] Chisel数据类型
- [ ] 基本模块编写
- [ ] 简单测试

### 第2周：进阶特性
- [ ] Bundle和Vec
- [ ] 状态机设计
- [ ] 存储器使用
- [ ] 参数化模块

### 第3周：实战项目
- [ ] 完整模块设计
- [ ] 复杂测试编写
- [ ] Verilog生成
- [ ] 调试技巧

### 第4周：高级主题
- [ ] 形式化验证
- [ ] 性能优化
- [ ] 工具链集成
- [ ] 团队协作

## 🔗 有用资源

- [Chisel官方文档](https://www.chisel-lang.org/)
- [Chisel Bootcamp](https://github.com/freechipsproject/chisel-bootcamp)
- [ChiselTest文档](https://github.com/ucb-bar/chiseltest)
- [Scala官方教程](https://docs.scala-lang.org/tour/tour-of-scala.html)

## 💡 小贴士

1. **从小模块开始** - 先掌握基本语法再写复杂逻辑
2. **多写测试** - 测试驱动开发，确保功能正确
3. **查看生成的Verilog** - 理解Chisel如何转换为硬件
4. **使用类型系统** - 让编译器帮你发现错误
5. **参考开源项目** - 学习最佳实践和设计模式