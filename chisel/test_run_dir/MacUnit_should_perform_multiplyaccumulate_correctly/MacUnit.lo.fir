FIRRTL version 1.2.0
circuit MacUnit :
  module MacUnit : @[src/main/scala/MacUnit.scala 9:7]
    input clock : Clock @[src/main/scala/MacUnit.scala 9:7]
    input reset : UInt<1> @[src/main/scala/MacUnit.scala 9:7]
    input io_a : SInt<32> @[src/main/scala/MacUnit.scala 10:14]
    input io_b : SInt<32> @[src/main/scala/MacUnit.scala 10:14]
    input io_c : SInt<32> @[src/main/scala/MacUnit.scala 10:14]
    output io_result : SInt<32> @[src/main/scala/MacUnit.scala 10:14]
    output io_valid : UInt<1> @[src/main/scala/MacUnit.scala 10:14]

    reg mult_result : SInt<32>, clock with :
      reset => (UInt<1>("h0"), mult_result) @[src/main/scala/MacUnit.scala 19:28]
    reg add_result : SInt<32>, clock with :
      reset => (UInt<1>("h0"), add_result) @[src/main/scala/MacUnit.scala 20:27]
    reg valid_reg : UInt<1>, clock with :
      reset => (UInt<1>("h0"), valid_reg) @[src/main/scala/MacUnit.scala 21:26]
    node _mult_result_T = mul(io_a, io_b) @[src/main/scala/MacUnit.scala 24:23]
    node _add_result_T = add(mult_result, io_c) @[src/main/scala/MacUnit.scala 27:29]
    node _add_result_T_1 = tail(_add_result_T, 1) @[src/main/scala/MacUnit.scala 27:29]
    node _add_result_T_2 = asSInt(_add_result_T_1) @[src/main/scala/MacUnit.scala 27:29]
    reg valid_pipe_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), valid_pipe_0) @[src/main/scala/MacUnit.scala 33:27]
    reg valid_pipe_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), valid_pipe_1) @[src/main/scala/MacUnit.scala 33:27]
    node _valid_pipe_WIRE_0 = UInt<1>("h0") @[src/main/scala/MacUnit.scala 33:{35,35}]
    node _valid_pipe_WIRE_1 = UInt<1>("h0") @[src/main/scala/MacUnit.scala 33:{35,35}]
    io_result <= add_result @[src/main/scala/MacUnit.scala 30:13]
    io_valid <= valid_pipe_1 @[src/main/scala/MacUnit.scala 36:12]
    mult_result <= asSInt(bits(mux(reset, asSInt(UInt<32>("h0")), _mult_result_T), 31, 0)) @[src/main/scala/MacUnit.scala 19:{28,28} 24:15]
    add_result <= mux(reset, asSInt(UInt<32>("h0")), _add_result_T_2) @[src/main/scala/MacUnit.scala 20:{27,27} 27:14]
    valid_reg <= mux(reset, UInt<1>("h0"), valid_reg) @[src/main/scala/MacUnit.scala 21:{26,26,26}]
    valid_pipe_0 <= mux(reset, _valid_pipe_WIRE_0, UInt<1>("h1")) @[src/main/scala/MacUnit.scala 33:{27,27} 34:17]
    valid_pipe_1 <= mux(reset, _valid_pipe_WIRE_1, valid_pipe_0) @[src/main/scala/MacUnit.scala 33:{27,27} 35:17]
