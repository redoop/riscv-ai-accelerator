FIRRTL version 1.2.0
circuit MacUnit :
  module MacUnit : @[src/main/scala/MatrixMultiplier.scala 10:7]
    input clock : Clock @[src/main/scala/MatrixMultiplier.scala 10:7]
    input reset : UInt<1> @[src/main/scala/MatrixMultiplier.scala 10:7]
    input io_a : SInt<32> @[src/main/scala/MatrixMultiplier.scala 11:14]
    input io_b : SInt<32> @[src/main/scala/MatrixMultiplier.scala 11:14]
    input io_c : SInt<32> @[src/main/scala/MatrixMultiplier.scala 11:14]
    output io_result : SInt<32> @[src/main/scala/MatrixMultiplier.scala 11:14]
    output io_valid : UInt<1> @[src/main/scala/MatrixMultiplier.scala 11:14]

    node product = mul(io_a, io_b) @[src/main/scala/MatrixMultiplier.scala 21:22]
    node _sum_T = add(product, io_c) @[src/main/scala/MatrixMultiplier.scala 22:21]
    node _sum_T_1 = tail(_sum_T, 1) @[src/main/scala/MatrixMultiplier.scala 22:21]
    node sum = asSInt(_sum_T_1) @[src/main/scala/MatrixMultiplier.scala 22:21]
    io_result <= asSInt(bits(sum, 31, 0)) @[src/main/scala/MatrixMultiplier.scala 25:13]
    io_valid <= UInt<1>("h1") @[src/main/scala/MatrixMultiplier.scala 26:12]
