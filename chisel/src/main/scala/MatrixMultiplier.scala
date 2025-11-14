package riscv.ai

import chisel3._
import chisel3.util._

/**
 * 矩阵存储器接口
 */
class MatrixMemoryIO(dataWidth: Int, addrWidth: Int) extends Bundle {
  val writeEn = Input(Bool())
  val readEn = Input(Bool())
  val addr = Input(UInt(addrWidth.W))
  val writeData = Input(SInt(dataWidth.W))
  val readData = Output(SInt(dataWidth.W))
}

/**
 * 矩阵乘法器
 * 执行 C = A × B，其中 A, B, C 都是 size×size 的矩阵
 */
class MatrixMultiplier(dataWidth: Int = 32, size: Int = 8) extends Module {
  val addrWidth = log2Ceil(size * size)
  
  val io = IO(new Bundle {
    val start = Input(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    
    val matrixA = new MatrixMemoryIO(dataWidth, addrWidth)
    val matrixB = new MatrixMemoryIO(dataWidth, addrWidth)
    val result = new MatrixMemoryIO(dataWidth, addrWidth)
  })
  
  // 矩阵存储器
  val memA = SyncReadMem(size * size, SInt(dataWidth.W))
  val memB = SyncReadMem(size * size, SInt(dataWidth.W))
  val memC = SyncReadMem(size * size, SInt(dataWidth.W))
  
  // 状态机
  val s_idle :: s_compute :: s_done :: Nil = Enum(3)
  val state = RegInit(s_idle)
  
  // 计算索引
  val row = RegInit(0.U(log2Ceil(size).W))
  val col = RegInit(0.U(log2Ceil(size).W))
  val k = RegInit(0.U(log2Ceil(size).W))
  
  // 累加器
  val accumulator = RegInit(0.S(dataWidth.W))
  
  // 读取的数据
  val dataA = RegInit(0.S(dataWidth.W))
  val dataB = RegInit(0.S(dataWidth.W))
  
  // 默认输出
  io.busy := state =/= s_idle
  io.done := state === s_done
  
  // 矩阵 A 接口
  when(io.matrixA.writeEn) {
    memA.write(io.matrixA.addr, io.matrixA.writeData)
  }
  io.matrixA.readData := memA.read(io.matrixA.addr, io.matrixA.readEn)
  
  // 矩阵 B 接口
  when(io.matrixB.writeEn) {
    memB.write(io.matrixB.addr, io.matrixB.writeData)
  }
  io.matrixB.readData := memB.read(io.matrixB.addr, io.matrixB.readEn)
  
  // 结果矩阵接口
  when(io.result.writeEn) {
    memC.write(io.result.addr, io.result.writeData)
  }
  io.result.readData := memC.read(io.result.addr, io.result.readEn)
  
  // 状态机
  switch(state) {
    is(s_idle) {
      when(io.start) {
        state := s_compute
        row := 0.U
        col := 0.U
        k := 0.U
        accumulator := 0.S
      }
    }
    
    is(s_compute) {
      // 读取 A[row][k] 和 B[k][col]
      val addrA = row * size.U + k
      val addrB = k * size.U + col
      
      dataA := memA.read(addrA, true.B)
      dataB := memB.read(addrB, true.B)
      
      // 累加
      accumulator := accumulator + dataA * dataB
      
      // 更新索引
      when(k === (size - 1).U) {
        // 写入结果
        val addrC = row * size.U + col
        memC.write(addrC, accumulator)
        
        k := 0.U
        accumulator := 0.S
        
        when(col === (size - 1).U) {
          col := 0.U
          when(row === (size - 1).U) {
            state := s_done
          }.otherwise {
            row := row + 1.U
          }
        }.otherwise {
          col := col + 1.U
        }
      }.otherwise {
        k := k + 1.U
      }
    }
    
    is(s_done) {
      when(!io.start) {
        state := s_idle
      }
    }
  }
}
