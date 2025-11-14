package riscv.ai

import chisel3._

/**
 * MAC (Multiply-Accumulate) 单元
 * 执行 result = a * b + c 操作
 */
class MacUnit(dataWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(dataWidth.W))
    val b = Input(SInt(dataWidth.W))
    val c = Input(SInt(dataWidth.W))
    val result = Output(SInt(dataWidth.W))
    val valid = Output(Bool())
  })
  
  // 流水线寄存器
  val mult_result = RegInit(0.S(dataWidth.W))
  val add_result = RegInit(0.S(dataWidth.W))
  val valid_reg = RegInit(false.B)
  
  // 第一级: 乘法
  mult_result := io.a * io.b
  
  // 第二级: 加法
  add_result := mult_result + io.c
  
  // 输出
  io.result := add_result
  
  // 有效信号 (延迟2个周期)
  val valid_pipe = RegInit(VecInit(Seq.fill(2)(false.B)))
  valid_pipe(0) := true.B
  valid_pipe(1) := valid_pipe(0)
  io.valid := valid_pipe(1)
}
