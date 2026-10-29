// RUN: arm64-mlir-opt %s -triton-arm64-linalg-tiling="vector-ext=sve" | FileCheck %s
// RUN: arm64-mlir-opt %s -triton-arm64-linalg-tiling="vector-ext=sme" | FileCheck %s --check-prefix=CHECK-SME

// CHECK-LABEL: func.func @test_elementwise_add
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
// Check that SME flat is ignored for non-matmul ops
// CHECK-SME-LABEL: func.func @test_elementwise_add
// CHECK-SME: [[VSCALE:%.+]] = vector.vscale
// CHECK-SME: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK-SME: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK-SME-NOT:   arith.constant 2
// CHECK-SME: scf.for {{.+}} step [[STEP_S]]

func.func @test_elementwise_add(%arg0: tensor<128xf32>,
                                 %arg1: tensor<128xf32>,
                                 %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>)
    outs(%arg2 : tensor<128xf32>)
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @test_reduce
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
func.func @test_reduce(%arg0: tensor<128xf32>,
                        %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.reduce { arith.addf }
    ins(%arg0 : tensor<128xf32>)
    outs(%arg1 : tensor<f32>)
    dimensions = [0]
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_matvec
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
// CHECK-SME-LABEL: func.func @test_matvec
// CHECK-SME: [[VSCALE:%.+]] = vector.vscale
// CHECK-SME: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK-SME: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK-SME: scf.for {{.+}} step [[STEP_S]]
// CHECK-SME-NOT: arith.constant 2
// CHECK-SME:     return
func.func @test_matvec(%arg0: tensor<64x128xf32>,
                        %arg1: tensor<128xf32>,
                        %arg2: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.matvec
    ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128xf32>)
    outs(%arg2 : tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_matmul
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
// CHECK-SME-LABEL: func.func @test_matmul
// CHECK-SME: [[VSCALE:%.+]] = vector.vscale
// CHECK-SME: [[C8_D0:%.+]] = arith.constant 8
// CHECK-SME: [[STEP_D0:%.+]] = arith.muli [[C8_D0]], [[VSCALE]]
// CHECK-SME: [[C8_D1:%.+]] = arith.constant 8
// CHECK-SME: [[STEP_D1:%.+]] = arith.muli [[C8_D1]], [[VSCALE]]
// CHECK-SME: [[CONST2:%.+]] = arith.constant 2
// CHECK-SME-NOT: arith.muli
// CHECK-SME: scf.for {{.+}} step [[STEP_D0]]
// CHECK-SME: scf.for {{.+}} step [[STEP_D1]]
// CHECK-SME: scf.for {{.+}} step [[CONST2]]
func.func @test_matmul(%arg0: tensor<64x128xf32>,
                        %arg1: tensor<128x64xf32>,
                        %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
    outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @test_no_tiling_memref
// CHECK-NOT: scf.for
func.func @test_no_tiling_memref(%arg0: memref<128xf32>,
                                  %arg1: memref<128xf32>,
                                  %arg2: memref<128xf32>) {
  linalg.map { arith.addf }
    ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>)
    outs(%arg2 : memref<128xf32>)
  return
}

// CHECK-LABEL: func.func @test_elementwise_2d
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[STEP_1:%.+]] = arith.constant 1
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_1]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
func.func @test_elementwise_2d(%arg0: tensor<32x128xf32>,
                                %arg1: tensor<32x128xf32>,
                                %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<32x128xf32>, tensor<32x128xf32>)
    outs(%arg2 : tensor<32x128xf32>)
  return %0 : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @test_generic_no_scalable
// CHECK-NOT:   vector.vscale
// CHECK:       arith.constant 1 : index
// CHECK:       arith.constant 8 : index
// CHECK:       scf.for
// CHECK:       scf.for
func.func @test_generic_no_scalable(%arg0: tensor<64x64xf32>,
                                     %arg1: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%arg0 : tensor<64x64xf32>)
  outs(%arg1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_mixed_tileable_and_memref
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
func.func @test_mixed_tileable_and_memref(%arg0: tensor<128xf32>,
                                           %arg1: tensor<128xf32>,
                                           %arg2: tensor<128xf32>,
                                           %arg3: memref<128xf32>,
                                           %arg4: memref<128xf32>,
                                           %arg5: memref<128xf32>) -> tensor<128xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>)
    outs(%arg2 : tensor<128xf32>)
  linalg.map { arith.addf }
    ins(%arg3, %arg4 : memref<128xf32>, memref<128xf32>)
    outs(%arg5 : memref<128xf32>)
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @test_no_tiling_zero_loops
// CHECK:       linalg.generic
// CHECK-NOT:   scf.for
func.func @test_no_tiling_zero_loops(%arg0: tensor<f32>,
                                      %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<() -> ()>,
      affine_map<() -> ()>
    ],
    iterator_types = []
  }
  ins(%arg0 : tensor<f32>)
  outs(%arg1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_conv1d_outer_product_tiling
// CHECK:       scf.for {{.+}} step {{.+}}4
// CHECK:       scf.for {{.+}} step {{.+}}4
// CHECK:       scf.for {{.+}} step {{.+}}1
// CHECK:       scf.for {{.+}} step {{.+}}1
// CHECK:       linalg.conv_1d_ncw_fcw
func.func @test_conv1d_outer_product_tiling(
    %input: tensor<1x3x6xf32>, %filter: tensor<4x3x3xf32>,
    %output: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %0 = linalg.conv_1d_ncw_fcw
    ins(%input, %filter : tensor<1x3x6xf32>, tensor<4x3x3xf32>)
    outs(%output : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}

// CHECK-LABEL: func.func @test_conv2d_decomposes_to_conv1d
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK-NOT:   linalg.conv_2d_nchw_fchw
// CHECK:       linalg.conv_1d_ncw_fcw
// CHECK:       return
func.func @test_conv2d_decomposes_to_conv1d(
    %input: tensor<1x3x5x6xf32>,
    %filter: tensor<4x3x3x3xf32>,
    %output: tensor<1x4x3x4xf32>) -> tensor<1x4x3x4xf32> {
  %0 = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<1x3x5x6xf32>, tensor<4x3x3x3xf32>)
    outs(%output : tensor<1x4x3x4xf32>) -> tensor<1x4x3x4xf32>
  return %0 : tensor<1x4x3x4xf32>
}
