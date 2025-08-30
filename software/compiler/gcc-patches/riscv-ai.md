;; RISC-V AI Extension Machine Description
;; Copyright (C) 2023 Free Software Foundation, Inc.

;; This file is part of GCC.

;; GCC is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 3, or (at your option)
;; any later version.

;; GCC is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with GCC; see the file COPYING3.  If not see
;; <http://www.gnu.org/licenses/>.

;; AI instruction patterns

;; Matrix multiplication instruction
(define_insn "ai_matmul_f32"
  [(unspec_volatile [(match_operand:DI 0 "register_operand" "r")  ; matrix A ptr
                     (match_operand:DI 1 "register_operand" "r")  ; matrix B ptr  
                     (match_operand:DI 2 "register_operand" "r")  ; matrix C ptr
                     (match_operand:DI 3 "register_operand" "r")] ; dimensions
                    UNSPEC_AI_MATMUL)]
  "TARGET_AI"
  "ai.matmul\t%2,%0,%1,%3"
  [(set_attr "type" "ai_matmul")
   (set_attr "mode" "DI")])

;; 2D Convolution instruction
(define_insn "ai_conv2d_f32"
  [(unspec_volatile [(match_operand:DI 0 "register_operand" "r")  ; input ptr
                     (match_operand:DI 1 "register_operand" "r")  ; kernel ptr
                     (match_operand:DI 2 "register_operand" "r")  ; output ptr
                     (match_operand:DI 3 "register_operand" "r")] ; parameters
                    UNSPEC_AI_CONV2D)]
  "TARGET_AI"
  "ai.conv2d\t%2,%0,%1,%3"
  [(set_attr "type" "ai_conv2d")
   (set_attr "mode" "DI")])

;; ReLU activation function
(define_insn "ai_relu_f32"
  [(set (match_operand:SF 0 "register_operand" "=f")
        (unspec:SF [(match_operand:SF 1 "register_operand" "f")]
                   UNSPEC_AI_RELU))]
  "TARGET_AI"
  "ai.relu\t%0,%1"
  [(set_attr "type" "ai_activation")
   (set_attr "mode" "SF")])

;; Sigmoid activation function
(define_insn "ai_sigmoid_f32"
  [(set (match_operand:SF 0 "register_operand" "=f")
        (unspec:SF [(match_operand:SF 1 "register_operand" "f")]
                   UNSPEC_AI_SIGMOID))]
  "TARGET_AI"
  "ai.sigmoid\t%0,%1"
  [(set_attr "type" "ai_activation")
   (set_attr "mode" "SF")])

;; Tanh activation function
(define_insn "ai_tanh_f32"
  [(set (match_operand:SF 0 "register_operand" "=f")
        (unspec:SF [(match_operand:SF 1 "register_operand" "f")]
                   UNSPEC_AI_TANH))]
  "TARGET_AI"
  "ai.tanh\t%0,%1"
  [(set_attr "type" "ai_activation")
   (set_attr "mode" "SF")])

;; Max pooling instruction
(define_insn "ai_maxpool_f32"
  [(unspec_volatile [(match_operand:DI 0 "register_operand" "r")  ; input ptr
                     (match_operand:DI 1 "register_operand" "r")  ; output ptr
                     (match_operand:DI 2 "register_operand" "r")] ; parameters
                    UNSPEC_AI_MAXPOOL)]
  "TARGET_AI"
  "ai.maxpool\t%1,%0,%2"
  [(set_attr "type" "ai_pooling")
   (set_attr "mode" "DI")])

;; Average pooling instruction
(define_insn "ai_avgpool_f32"
  [(unspec_volatile [(match_operand:DI 0 "register_operand" "r")  ; input ptr
                     (match_operand:DI 1 "register_operand" "r")  ; output ptr
                     (match_operand:DI 2 "register_operand" "r")] ; parameters
                    UNSPEC_AI_AVGPOOL)]
  "TARGET_AI"
  "ai.avgpool\t%1,%0,%2"
  [(set_attr "type" "ai_pooling")
   (set_attr "mode" "DI")])

;; Batch normalization instruction
(define_insn "ai_batchnorm_f32"
  [(unspec_volatile [(match_operand:DI 0 "register_operand" "r")  ; input ptr
                     (match_operand:DI 1 "register_operand" "r")  ; output ptr
                     (match_operand:DI 2 "register_operand" "r")  ; scale ptr
                     (match_operand:DI 3 "register_operand" "r")] ; bias ptr
                    UNSPEC_AI_BATCHNORM)]
  "TARGET_AI"
  "ai.batchnorm\t%1,%0,%2,%3"
  [(set_attr "type" "ai_batchnorm")
   (set_attr "mode" "DI")])

;; AI accelerator status read
(define_insn "ai_get_status"
  [(set (match_operand:SI 0 "register_operand" "=r")
        (unspec:SI [(match_operand:SI 1 "register_operand" "r")]
                   UNSPEC_AI_GET_STATUS))]
  "TARGET_AI"
  "csrr\t%0,0x800"
  [(set_attr "type" "csr")
   (set_attr "mode" "SI")])

;; AI accelerator configuration write
(define_insn "ai_set_config"
  [(unspec_volatile [(match_operand:SI 0 "register_operand" "r")  ; accel_id
                     (match_operand:SI 1 "register_operand" "r")] ; config
                    UNSPEC_AI_SET_CONFIG)]
  "TARGET_AI"
  "csrw\t0x801,%1"
  [(set_attr "type" "csr")
   (set_attr "mode" "SI")])

;; AI accelerator pipeline flush
(define_insn "ai_flush"
  [(unspec_volatile [(match_operand:SI 0 "register_operand" "r")]
                    UNSPEC_AI_FLUSH)]
  "TARGET_AI"
  "ai.flush\t%0"
  [(set_attr "type" "ai_control")
   (set_attr "mode" "SI")])

;; AI accelerator synchronization
(define_insn "ai_sync"
  [(unspec_volatile [(match_operand:SI 0 "register_operand" "r")]
                    UNSPEC_AI_SYNC)]
  "TARGET_AI"
  "ai.sync\t%0"
  [(set_attr "type" "ai_control")
   (set_attr "mode" "SI")])

;; AI instruction scheduling and pipeline definitions
(define_cpu_unit "ai_matmul_unit" "riscv_ai")
(define_cpu_unit "ai_conv_unit" "riscv_ai")
(define_cpu_unit "ai_activation_unit" "riscv_ai")
(define_cpu_unit "ai_pooling_unit" "riscv_ai")

;; AI instruction reservations
(define_insn_reservation "ai_matmul_insn" 10
  (eq_attr "type" "ai_matmul")
  "ai_matmul_unit*10")

(define_insn_reservation "ai_conv2d_insn" 8
  (eq_attr "type" "ai_conv2d")
  "ai_conv_unit*8")

(define_insn_reservation "ai_activation_insn" 2
  (eq_attr "type" "ai_activation")
  "ai_activation_unit*2")

(define_insn_reservation "ai_pooling_insn" 4
  (eq_attr "type" "ai_pooling")
  "ai_pooling_unit*4")

(define_insn_reservation "ai_batchnorm_insn" 6
  (eq_attr "type" "ai_batchnorm")
  "ai_activation_unit*6")

(define_insn_reservation "ai_control_insn" 1
  (eq_attr "type" "ai_control")
  "nothing")

;; AI optimization patterns

;; Combine multiple ReLU operations
(define_peephole2
  [(set (match_operand:SF 0 "register_operand" "")
        (unspec:SF [(match_operand:SF 1 "register_operand" "")]
                   UNSPEC_AI_RELU))
   (set (match_operand:SF 2 "register_operand" "")
        (unspec:SF [(match_dup 0)]
                   UNSPEC_AI_RELU))]
  "TARGET_AI && peep2_reg_dead_p (2, operands[0])"
  [(set (match_dup 2)
        (unspec:SF [(match_dup 1)]
                   UNSPEC_AI_RELU))])

;; Fuse matrix multiplication with activation
(define_peephole2
  [(unspec_volatile [(match_operand:DI 0 "register_operand" "")
                     (match_operand:DI 1 "register_operand" "")
                     (match_operand:DI 2 "register_operand" "")
                     (match_operand:DI 3 "register_operand" "")]
                    UNSPEC_AI_MATMUL)
   (parallel [(set (match_operand:SF 4 "register_operand" "")
                   (unspec:SF [(mem:SF (match_dup 2))]
                              UNSPEC_AI_RELU))
              (clobber (match_scratch:SI 5 ""))])]
  "TARGET_AI"
  [(unspec_volatile [(match_dup 0) (match_dup 1) (match_dup 2) 
                     (ior:DI (match_dup 3) (const_int 0x10000))]
                    UNSPEC_AI_MATMUL_RELU)])

;; Vector load/store optimization for AI data
(define_peephole2
  [(set (match_operand:SF 0 "register_operand" "")
        (mem:SF (plus:DI (match_operand:DI 1 "register_operand" "")
                         (match_operand:DI 2 "const_int_operand" ""))))
   (set (match_operand:SF 3 "register_operand" "")
        (mem:SF (plus:DI (match_dup 1)
                         (match_operand:DI 4 "const_int_operand" ""))))]
  "TARGET_AI && TARGET_VECTOR
   && INTVAL (operands[4]) == INTVAL (operands[2]) + 4"
  [(parallel [(set (match_dup 0) (mem:SF (plus:DI (match_dup 1) (match_dup 2))))
              (set (match_dup 3) (mem:SF (plus:DI (match_dup 1) (match_dup 4))))])])