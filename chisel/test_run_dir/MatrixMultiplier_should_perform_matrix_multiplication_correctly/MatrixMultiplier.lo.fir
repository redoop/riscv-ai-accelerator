FIRRTL version 1.2.0
circuit MatrixMultiplier :
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

  module MatrixMultiplier : @[src/main/scala/MatrixMultiplier.scala 33:7]
    input clock : Clock @[src/main/scala/MatrixMultiplier.scala 33:7]
    input reset : UInt<1> @[src/main/scala/MatrixMultiplier.scala 33:7]
    input io_start : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    output io_done : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    output io_busy : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixA_writeEn : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixA_readEn : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixA_addr : UInt<2> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixA_writeData : SInt<32> @[src/main/scala/MatrixMultiplier.scala 39:14]
    output io_matrixA_readData : SInt<32> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixB_writeEn : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixB_readEn : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixB_addr : UInt<2> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_matrixB_writeData : SInt<32> @[src/main/scala/MatrixMultiplier.scala 39:14]
    output io_matrixB_readData : SInt<32> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_result_writeEn : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_result_readEn : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_result_addr : UInt<2> @[src/main/scala/MatrixMultiplier.scala 39:14]
    input io_result_writeData : SInt<32> @[src/main/scala/MatrixMultiplier.scala 39:14]
    output io_result_readData : SInt<32> @[src/main/scala/MatrixMultiplier.scala 39:14]
    output io_result_valid : UInt<1> @[src/main/scala/MatrixMultiplier.scala 39:14]

    mem matrixA : @[src/main/scala/MatrixMultiplier.scala 75:20]
      data-type => SInt<32>
      depth => 4
      read-latency => 0
      write-latency => 1
      reader => io_matrixA_readData_MPORT
      reader => macUnit_io_a_MPORT
      writer => MPORT
      read-under-write => undefined
    mem matrixB : @[src/main/scala/MatrixMultiplier.scala 76:20]
      data-type => SInt<32>
      depth => 4
      read-latency => 0
      write-latency => 1
      reader => io_matrixB_readData_MPORT
      reader => macUnit_io_b_MPORT
      writer => MPORT_1
      read-under-write => undefined
    mem matrixResult : @[src/main/scala/MatrixMultiplier.scala 77:25]
      data-type => SInt<32>
      depth => 4
      read-latency => 0
      write-latency => 1
      reader => io_result_readData_MPORT
      writer => MPORT_2
      writer => MPORT_3
      read-under-write => undefined
    inst macUnit of MacUnit @[src/main/scala/MatrixMultiplier.scala 80:23]
    reg state : UInt<2>, clock with :
      reset => (UInt<1>("h0"), state) @[src/main/scala/MatrixMultiplier.scala 84:22]
    reg cycleCounter : UInt<4>, clock with :
      reset => (UInt<1>("h0"), cycleCounter) @[src/main/scala/MatrixMultiplier.scala 88:29]
    reg computationComplete : UInt<1>, clock with :
      reset => (UInt<1>("h0"), computationComplete) @[src/main/scala/MatrixMultiplier.scala 89:36]
    node k = rem(cycleCounter, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 93:24]
    node temp = div(cycleCounter, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 94:27]
    node j = rem(temp, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 95:16]
    node i = div(temp, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 96:16]
    reg iDelayed : UInt<4>, clock with :
      reset => (UInt<1>("h0"), iDelayed) @[src/main/scala/MatrixMultiplier.scala 99:25]
    reg jDelayed : UInt<2>, clock with :
      reset => (UInt<1>("h0"), jDelayed) @[src/main/scala/MatrixMultiplier.scala 100:25]
    reg kDelayed : UInt<2>, clock with :
      reset => (UInt<1>("h0"), kDelayed) @[src/main/scala/MatrixMultiplier.scala 101:25]
    reg accValue : SInt<32>, clock with :
      reset => (UInt<1>("h0"), accValue) @[src/main/scala/MatrixMultiplier.scala 104:25]
    node _GEN_0 = validif(io_matrixA_writeEn, io_matrixA_addr) @[src/main/scala/MatrixMultiplier.scala 108:28 109:12]
    node _GEN_1 = validif(io_matrixA_writeEn, clock) @[src/main/scala/MatrixMultiplier.scala 108:28 109:12]
    node _GEN_2 = mux(io_matrixA_writeEn, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 108:28 109:12 75:20]
    node _GEN_3 = validif(io_matrixA_writeEn, UInt<1>("h1")) @[src/main/scala/MatrixMultiplier.scala 108:28 109:30]
    node _GEN_4 = validif(io_matrixA_writeEn, io_matrixA_writeData) @[src/main/scala/MatrixMultiplier.scala 108:28 109:30]
    node _io_matrixA_readData_T = mux(io_matrixA_readEn, matrixA.io_matrixA_readData_MPORT.data, asSInt(UInt<1>("h0"))) @[src/main/scala/MatrixMultiplier.scala 111:29]
    node _GEN_5 = validif(io_matrixB_writeEn, io_matrixB_addr) @[src/main/scala/MatrixMultiplier.scala 114:28 115:12]
    node _GEN_6 = validif(io_matrixB_writeEn, clock) @[src/main/scala/MatrixMultiplier.scala 114:28 115:12]
    node _GEN_7 = mux(io_matrixB_writeEn, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 114:28 115:12 76:20]
    node _GEN_8 = validif(io_matrixB_writeEn, UInt<1>("h1")) @[src/main/scala/MatrixMultiplier.scala 114:28 115:30]
    node _GEN_9 = validif(io_matrixB_writeEn, io_matrixB_writeData) @[src/main/scala/MatrixMultiplier.scala 114:28 115:30]
    node _io_matrixB_readData_T = mux(io_matrixB_readEn, matrixB.io_matrixB_readData_MPORT.data, asSInt(UInt<1>("h0"))) @[src/main/scala/MatrixMultiplier.scala 117:29]
    node _GEN_10 = validif(io_result_writeEn, io_result_addr) @[src/main/scala/MatrixMultiplier.scala 120:27 121:17]
    node _GEN_11 = validif(io_result_writeEn, clock) @[src/main/scala/MatrixMultiplier.scala 120:27 121:17]
    node _GEN_12 = mux(io_result_writeEn, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 120:27 121:17 77:25]
    node _GEN_13 = validif(io_result_writeEn, UInt<1>("h1")) @[src/main/scala/MatrixMultiplier.scala 120:27 121:34]
    node _GEN_14 = validif(io_result_writeEn, io_result_writeData) @[src/main/scala/MatrixMultiplier.scala 120:27 121:34]
    node _io_result_readData_T = mux(io_result_readEn, matrixResult.io_result_readData_MPORT.data, asSInt(UInt<1>("h0"))) @[src/main/scala/MatrixMultiplier.scala 124:28]
    node _T = eq(UInt<2>("h0"), state) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _GEN_15 = mux(io_start, UInt<2>("h1"), state) @[src/main/scala/MatrixMultiplier.scala 130:22 131:15 84:22]
    node _GEN_16 = mux(io_start, UInt<1>("h0"), cycleCounter) @[src/main/scala/MatrixMultiplier.scala 130:22 132:22 88:29]
    node _GEN_17 = mux(io_start, UInt<1>("h0"), computationComplete) @[src/main/scala/MatrixMultiplier.scala 130:22 133:29 89:36]
    node _GEN_18 = mux(io_start, asSInt(UInt<1>("h0")), accValue) @[src/main/scala/MatrixMultiplier.scala 130:22 134:18 104:25]
    node _T_1 = eq(UInt<2>("h1"), state) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _cycleCounter_T = add(cycleCounter, UInt<1>("h1")) @[src/main/scala/MatrixMultiplier.scala 139:36]
    node _cycleCounter_T_1 = tail(_cycleCounter_T, 1) @[src/main/scala/MatrixMultiplier.scala 139:36]
    node _T_2 = eq(cycleCounter, UInt<3>("h7")) @[src/main/scala/MatrixMultiplier.scala 140:25]
    node _GEN_19 = mux(_T_2, UInt<1>("h1"), computationComplete) @[src/main/scala/MatrixMultiplier.scala 140:50 141:29 89:36]
    node _GEN_20 = mux(_T_2, UInt<2>("h2"), state) @[src/main/scala/MatrixMultiplier.scala 140:50 142:15 84:22]
    node _T_3 = eq(UInt<2>("h2"), state) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _T_4 = eq(io_start, UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 147:12]
    node _GEN_21 = mux(_T_4, UInt<2>("h0"), state) @[src/main/scala/MatrixMultiplier.scala 147:23 148:15 84:22]
    node _GEN_22 = mux(_T_3, _GEN_21, state) @[src/main/scala/MatrixMultiplier.scala 128:17 84:22]
    node _GEN_23 = mux(_T_1, _cycleCounter_T_1, cycleCounter) @[src/main/scala/MatrixMultiplier.scala 128:17 139:20 88:29]
    node _GEN_24 = mux(_T_1, _GEN_19, computationComplete) @[src/main/scala/MatrixMultiplier.scala 128:17 89:36]
    node _GEN_25 = mux(_T_1, _GEN_20, _GEN_22) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _GEN_26 = mux(_T, _GEN_15, _GEN_25) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _GEN_27 = mux(_T, _GEN_16, _GEN_23) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _GEN_28 = mux(_T, _GEN_17, _GEN_24) @[src/main/scala/MatrixMultiplier.scala 128:17]
    node _GEN_29 = mux(_T, _GEN_18, accValue) @[src/main/scala/MatrixMultiplier.scala 128:17 104:25]
    node _aAddr_T = mul(i, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 154:17]
    node _aAddr_T_1 = add(_aAddr_T, k) @[src/main/scala/MatrixMultiplier.scala 154:32]
    node aAddr = tail(_aAddr_T_1, 1) @[src/main/scala/MatrixMultiplier.scala 154:32]
    node _bAddr_T = mul(k, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 155:17]
    node _bAddr_T_1 = add(_bAddr_T, j) @[src/main/scala/MatrixMultiplier.scala 155:32]
    node bAddr = tail(_bAddr_T_1, 1) @[src/main/scala/MatrixMultiplier.scala 155:32]
    node _cAddr_T = mul(i, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 156:17]
    node _cAddr_T_1 = add(_cAddr_T, j) @[src/main/scala/MatrixMultiplier.scala 156:32]
    node cAddr = tail(_cAddr_T_1, 1) @[src/main/scala/MatrixMultiplier.scala 156:32]
    node _macUnit_io_a_T = bits(aAddr, 1, 0) @[src/main/scala/MatrixMultiplier.scala 158:26]
    node _macUnit_io_b_T = bits(bAddr, 1, 0) @[src/main/scala/MatrixMultiplier.scala 159:26]
    node _T_5 = eq(state, UInt<2>("h1")) @[src/main/scala/MatrixMultiplier.scala 162:14]
    node _T_6 = eq(k, UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 163:12]
    node _GEN_30 = mux(_T_6, asSInt(UInt<1>("h0")), accValue) @[src/main/scala/MatrixMultiplier.scala 163:21 165:20 169:20]
    node _GEN_31 = mux(_T_6, macUnit.io_result, macUnit.io_result) @[src/main/scala/MatrixMultiplier.scala 163:21 166:20 170:20]
    node _T_7 = eq(k, UInt<1>("h1")) @[src/main/scala/MatrixMultiplier.scala 177:12]
    node _resultAddr_T = mul(i, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 178:26]
    node _resultAddr_T_1 = add(_resultAddr_T, j) @[src/main/scala/MatrixMultiplier.scala 178:41]
    node resultAddr = tail(_resultAddr_T_1, 1) @[src/main/scala/MatrixMultiplier.scala 178:41]
    node _T_8 = bits(resultAddr, 1, 0) @[src/main/scala/MatrixMultiplier.scala 179:19]
    node _GEN_32 = validif(_T_7, _T_8) @[src/main/scala/MatrixMultiplier.scala 177:36 179:19]
    node _GEN_33 = validif(_T_7, clock) @[src/main/scala/MatrixMultiplier.scala 177:36 179:19]
    node _GEN_34 = mux(_T_7, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 177:36 179:19 77:25]
    node _GEN_35 = validif(_T_7, UInt<1>("h1")) @[src/main/scala/MatrixMultiplier.scala 177:36 179:32]
    node _GEN_38 = mux(_T_5, _GEN_31, asSInt(UInt<1>("h0"))) @[src/main/scala/MatrixMultiplier.scala 162:28 183:18]
    node accValueNext = _GEN_38 @[src/main/scala/MatrixMultiplier.scala 105:26]
    node _GEN_36 = validif(_T_7, accValueNext) @[src/main/scala/MatrixMultiplier.scala 177:36 179:32]
    node _GEN_37 = mux(_T_5, _GEN_30, asSInt(UInt<1>("h0"))) @[src/main/scala/MatrixMultiplier.scala 162:28 182:18]
    node _GEN_39 = mux(_T_5, accValueNext, _GEN_29) @[src/main/scala/MatrixMultiplier.scala 162:28 174:14]
    node _GEN_40 = validif(_T_5, _GEN_32) @[src/main/scala/MatrixMultiplier.scala 162:28]
    node _GEN_41 = validif(_T_5, _GEN_33) @[src/main/scala/MatrixMultiplier.scala 162:28]
    node _GEN_42 = mux(_T_5, _GEN_34, UInt<1>("h0")) @[src/main/scala/MatrixMultiplier.scala 162:28 77:25]
    node _GEN_43 = validif(_T_5, _GEN_35) @[src/main/scala/MatrixMultiplier.scala 162:28]
    node _GEN_44 = validif(_T_5, _GEN_36) @[src/main/scala/MatrixMultiplier.scala 162:28]
    node _io_busy_T = neq(state, UInt<2>("h0")) @[src/main/scala/MatrixMultiplier.scala 187:20]
    node _io_done_T = eq(state, UInt<2>("h2")) @[src/main/scala/MatrixMultiplier.scala 188:20]
    io_done <= _io_done_T @[src/main/scala/MatrixMultiplier.scala 188:11]
    io_busy <= _io_busy_T @[src/main/scala/MatrixMultiplier.scala 187:11]
    io_matrixA_readData <= _io_matrixA_readData_T @[src/main/scala/MatrixMultiplier.scala 111:23]
    io_matrixB_readData <= _io_matrixB_readData_T @[src/main/scala/MatrixMultiplier.scala 117:23]
    io_result_readData <= _io_result_readData_T @[src/main/scala/MatrixMultiplier.scala 124:22]
    io_result_valid <= io_result_readEn @[src/main/scala/MatrixMultiplier.scala 125:19]
    matrixA.io_matrixA_readData_MPORT.addr <= io_matrixA_addr @[src/main/scala/MatrixMultiplier.scala 111:56]
    matrixA.io_matrixA_readData_MPORT.en <= UInt<1>("h1") @[src/main/scala/MatrixMultiplier.scala 111:56]
    matrixA.io_matrixA_readData_MPORT.clk <= clock @[src/main/scala/MatrixMultiplier.scala 111:56]
    matrixA.macUnit_io_a_MPORT.addr <= _macUnit_io_a_T @[src/main/scala/MatrixMultiplier.scala 158:26]
    matrixA.macUnit_io_a_MPORT.en <= UInt<1>("h1") @[src/main/scala/MatrixMultiplier.scala 158:26]
    matrixA.macUnit_io_a_MPORT.clk <= clock @[src/main/scala/MatrixMultiplier.scala 158:26]
    matrixA.MPORT.addr <= _GEN_0
    matrixA.MPORT.en <= _GEN_2
    matrixA.MPORT.clk <= _GEN_1
    matrixA.MPORT.data <= _GEN_4
    matrixA.MPORT.mask <= _GEN_3
    matrixB.io_matrixB_readData_MPORT.addr <= io_matrixB_addr @[src/main/scala/MatrixMultiplier.scala 117:56]
    matrixB.io_matrixB_readData_MPORT.en <= UInt<1>("h1") @[src/main/scala/MatrixMultiplier.scala 117:56]
    matrixB.io_matrixB_readData_MPORT.clk <= clock @[src/main/scala/MatrixMultiplier.scala 117:56]
    matrixB.macUnit_io_b_MPORT.addr <= _macUnit_io_b_T @[src/main/scala/MatrixMultiplier.scala 159:26]
    matrixB.macUnit_io_b_MPORT.en <= UInt<1>("h1") @[src/main/scala/MatrixMultiplier.scala 159:26]
    matrixB.macUnit_io_b_MPORT.clk <= clock @[src/main/scala/MatrixMultiplier.scala 159:26]
    matrixB.MPORT_1.addr <= _GEN_5
    matrixB.MPORT_1.en <= _GEN_7
    matrixB.MPORT_1.clk <= _GEN_6
    matrixB.MPORT_1.data <= _GEN_9
    matrixB.MPORT_1.mask <= _GEN_8
    matrixResult.io_result_readData_MPORT.addr <= io_result_addr @[src/main/scala/MatrixMultiplier.scala 124:59]
    matrixResult.io_result_readData_MPORT.en <= UInt<1>("h1") @[src/main/scala/MatrixMultiplier.scala 124:59]
    matrixResult.io_result_readData_MPORT.clk <= clock @[src/main/scala/MatrixMultiplier.scala 124:59]
    matrixResult.MPORT_2.addr <= _GEN_10
    matrixResult.MPORT_2.en <= _GEN_12
    matrixResult.MPORT_2.clk <= _GEN_11
    matrixResult.MPORT_2.data <= _GEN_14
    matrixResult.MPORT_2.mask <= _GEN_13
    matrixResult.MPORT_3.addr <= _GEN_40
    matrixResult.MPORT_3.en <= _GEN_42
    matrixResult.MPORT_3.clk <= _GEN_41
    matrixResult.MPORT_3.data <= _GEN_44
    matrixResult.MPORT_3.mask <= _GEN_43
    macUnit.clock <= clock
    macUnit.reset <= reset
    macUnit.io_a <= matrixA.macUnit_io_a_MPORT.data @[src/main/scala/MatrixMultiplier.scala 158:16]
    macUnit.io_b <= matrixB.macUnit_io_b_MPORT.data @[src/main/scala/MatrixMultiplier.scala 159:16]
    macUnit.io_c <= _GEN_37
    state <= mux(reset, UInt<2>("h0"), _GEN_26) @[src/main/scala/MatrixMultiplier.scala 84:{22,22}]
    cycleCounter <= mux(reset, UInt<4>("h0"), _GEN_27) @[src/main/scala/MatrixMultiplier.scala 88:{29,29}]
    computationComplete <= mux(reset, UInt<1>("h0"), _GEN_28) @[src/main/scala/MatrixMultiplier.scala 89:{36,36}]
    iDelayed <= i @[src/main/scala/MatrixMultiplier.scala 99:25]
    jDelayed <= j @[src/main/scala/MatrixMultiplier.scala 100:25]
    kDelayed <= k @[src/main/scala/MatrixMultiplier.scala 101:25]
    accValue <= mux(reset, asSInt(UInt<32>("h0")), _GEN_39) @[src/main/scala/MatrixMultiplier.scala 104:{25,25}]
