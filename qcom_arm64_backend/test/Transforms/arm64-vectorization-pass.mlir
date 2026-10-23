// RUN: arm64-mlir-opt %s -triton-arm64-linalg-vectorization="vector-ext=neon" | FileCheck %s
// RUN: arm64-mlir-opt %s -triton-arm64-linalg-vectorization | FileCheck %s --check-prefix=CHECK-SME
// RUN: arm64-mlir-opt %s -triton-arm64-linalg-tiling="vector-ext=neon" -triton-arm64-linalg-vectorization="vector-ext=neon" | FileCheck %s --check-prefix=CHECK-CONV

// CHECK-LABEL: func.func @test_elementwise_static
// CHECK-NOT:   linalg.map
// CHECK:       [[V1:%.+]] = vector.transfer_read {{.+}} : tensor<8xf32>, vector<8xf32>
// CHECK:       [[V2:%.+]] = vector.transfer_read {{.+}} : tensor<8xf32>, vector<8xf32>
// CHECK:       [[ADD:%.+]] = arith.addf [[V1]], [[V2]] : vector<8xf32>
// CHECK:       vector.transfer_write [[ADD]], {{.+}} : vector<8xf32>, tensor<8xf32>
func.func @test_elementwise_static(%arg0: tensor<8xf32>,
                                    %arg1: tensor<8xf32>,
                                    %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%arg2 : tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_reduce_static
// CHECK-NOT:   linalg.reduce
// CHECK:       [[V1:%.+]] = vector.transfer_read {{.+}} : tensor<8xf32>, vector<8xf32>
// CHECK:       [[V2:%.+]] = vector.transfer_read {{.+}} : tensor<f32>, vector<f32>
// CHECK:       [[EXTRACT:%.+]] = vector.extract [[V2]]{{.+}} : f32 from vector<f32>
// CHECK:       [[MULTI_RED:%.+]] = vector.multi_reduction <add>, [[V1]], [[EXTRACT]] {{.+}} : vector<8xf32> to f32
// CHECK:       [[BROADCAST:%.+]] = vector.broadcast [[MULTI_RED]] : f32 to vector<f32>
// CHECK:       vector.transfer_write [[BROADCAST]], {{.+}} : vector<f32>, tensor<f32>
func.func @test_reduce_static(%arg0: tensor<8xf32>,
                               %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.reduce { arith.addf }
    ins(%arg0 : tensor<8xf32>)
    outs(%arg1 : tensor<f32>)
    dimensions = [0]
  return %0 : tensor<f32>
}

// i think matvec should get vectorized, but it currently doesn't
// CHECK-LABEL: func.func @test_matvec_dynamic
// CHECK:       linalg.matvec
// CHECK-NOT:   vector.transfer_read
func.func @test_matvec_dynamic(%arg0: tensor<4x?xf32>,
                                %arg1: tensor<?xf32>,
                                %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.matvec
    ins(%arg0, %arg1 : tensor<4x?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// currently, non-isSME path does not get vectorized since vecSizes = [1, 1, 4]
// CHECK-LABEL:    func.func @test_matmul_sme
// CHECK:          linalg.matmul
// CHECK-NOT:      vector.transfer_read
// CHECK-SME-LABEL: func.func @test_matmul_sme
// CHECK-SME-NOT:   linalg.matmul
// CHECK-SME-NOT:   vector<4x4xf32>
// CHECK-SME:       [[MASK1:%.+]] = vector.create_mask {{.+}} : vector<[8]x2xi1>
// CHECK-SME:       [[V1:%.+]] = vector.mask [[MASK1]] { vector.transfer_read {{.+}} : tensor<4x?xf32>, vector<[8]x[8]x2xf32> } : vector<[8]x2xi1> -> vector<[8]x[8]x2xf32>
// CHECK-SME:       [[MASK2:%.+]] = vector.create_mask {{.+}} : vector<2x[8]xi1>
// CHECK-SME:       [[V2:%.+]] = vector.mask [[MASK2]] { vector.transfer_read {{.+}} : tensor<?x4xf32>, vector<[8]x[8]x2xf32> } : vector<2x[8]xi1> -> vector<[8]x[8]x2xf32>
// CHECK-SME:       [[MASK3:%.+]] = vector.create_mask {{.+}} : vector<[8]x[8]xi1>
// CHECK-SME:       [[V3:%.+]] = vector.mask [[MASK3]] { vector.transfer_read {{.+}} : tensor<4x4xf32>, vector<[8]x[8]xf32> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xf32>
// CHECK-SME:       [[MUL:%.+]] = arith.mulf [[V1]], [[V2]] : vector<[8]x[8]x2xf32>
// CHECK-SME:       [[MASK4:%.+]] = vector.create_mask {{.+}} : vector<[8]x[8]x2xi1>
// CHECK-SME:       [[MULTI_RED:%.+]] = vector.mask [[MASK4]] { vector.multi_reduction <add>, [[MUL]], [[V3]] {{.+}} : vector<[8]x[8]x2xf32> to vector<[8]x[8]xf32> } : vector<[8]x[8]x2xi1> -> vector<[8]x[8]xf32>
// CHECK-SME:       vector.mask [[MASK3]] { vector.transfer_write [[MULTI_RED]], {{.+}} : vector<[8]x[8]xf32>, tensor<4x4xf32> }
func.func @test_matmul_sme(%arg0: tensor<4x?xf32>,
                            %arg1: tensor<?x4xf32>,
                            %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<4x?xf32>, tensor<?x4xf32>)
    outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_generic_parallel_static
// CHECK-NOT:   linalg.generic
// CHECK:       [[V1:%.+]] = vector.transfer_read {{.+}} : tensor<8xf32>, vector<8xf32>
// CHECK:       [[V2:%.+]] = vector.transfer_read {{.+}} : tensor<8xf32>, vector<8xf32>
// CHECK:       [[MUL:%.+]] = arith.mulf [[V1]], [[V1]] : vector<8xf32>
// CHECK:       vector.transfer_write [[MUL]], {{.+}} : vector<8xf32>, tensor<8xf32>
func.func @test_generic_parallel_static(%arg0: tensor<8xf32>,
                                         %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%arg0 : tensor<8xf32>)
  outs(%arg1 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %in, %in : f32
      linalg.yield %scaled : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_elementwise_dynamic_scalable
// CHECK:   linalg.map
func.func @test_elementwise_dynamic_scalable(%arg0: tensor<?xf32>,
                                              %arg1: tensor<?xf32>,
                                              %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>)
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @test_reduce_dynamic_scalable
// CHECK:   linalg.reduce
func.func @test_reduce_dynamic_scalable(%arg0: tensor<?xf32>,
                                         %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.reduce { arith.addf }
    ins(%arg0 : tensor<?xf32>)
    outs(%arg1 : tensor<f32>)
    dimensions = [0]
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_generic_dynamic_scalable
// CHECK:   linalg.generic
func.func @test_generic_dynamic_scalable(%arg0: tensor<?xf32>,
                                          %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%arg0 : tensor<?xf32>)
  outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %in, %in : f32
      linalg.yield %scaled : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @test_no_vectorize_zero_loops
// CHECK:       linalg.generic
// CHECK-NOT:   vector.transfer_read
func.func @test_no_vectorize_zero_loops(%arg0: tensor<f32>,
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

// CHECK-LABEL: func.func @test_library_call_skipped_matmul
// CHECK:       linalg.matmul
// CHECK-NOT:   vector.transfer_read
func.func @test_library_call_skipped_matmul(%arg0: tensor<4x8xf32>,
                                      %arg1: tensor<8x4xf32>,
                                      %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul {library_call = "foo"}
    ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x4xf32>)
    outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_library_call_skipped_generic
// CHECK:       linalg.generic
// CHECK-NOT:   vector.transfer_read
// CHECK:       return
func.func @test_library_call_skipped_generic(%arg0: tensor<8xf32>,
                                          %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"],
    library_call = "foo"
  }
  ins(%arg0 : tensor<8xf32>)
  outs(%arg1 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-CONV-LABEL: func.func @test_conv1d_outer_product
// CHECK-CONV:       scf.for
// CHECK-CONV:       vector.transfer_read {{.+}} : tensor<1x1x4xf32>, vector<1x1x4xf32>
// CHECK-CONV:       vector.transfer_read {{.+}} : tensor<4x1x1xf32>, vector<4x1x1xf32>
// CHECK-CONV:       vector.transfer_read {{.+}} : tensor<1x4x4xf32>, vector<1x4x4xf32>
// CHECK-CONV:       vector.contract
// CHECK-CONV:       vector.transfer_write {{.+}} : vector<1x4x4xf32>, tensor<1x4x4xf32>
// CHECK-CONV-NOT:   linalg.conv_1d_ncw_fcw
func.func @test_conv1d_outer_product(%input: tensor<1x3x6xf32>,
                                     %filter: tensor<4x3x3xf32>,
                                     %output: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %0 = linalg.conv_1d_ncw_fcw
    ins(%input, %filter : tensor<1x3x6xf32>, tensor<4x3x3xf32>)
    outs(%output : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}
