// RUN: arm64-mlir-opt %s -triton-arm64-reduction-scheduling="vector-ext=sve" | FileCheck %s
// RUN: arm64-mlir-opt %s -triton-arm64-reduction-scheduling="vector-ext=neon" | FileCheck %s --check-prefix=CHECK-NEON

// The SVE schedule uses four independent loop-carried accumulators.  Its
// cloned reductions and final combines are reassociable because the schedule
// intentionally changes the floating-point addition order.
// CHECK-LABEL: func.func @scheduled_masked_add_reduction
// CHECK:       %[[STEP:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:       %[[UNROLLED_STEP:.*]] = arith.muli %[[STEP]], %{{.*}} : index
// CHECK:       %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK:       %[[SCHEDULED:.*]]:4 = scf.for {{.*}} step %[[UNROLLED_STEP]] iter_args(%[[ACC0:.*]] = %arg2, %[[ACC1:.*]] = %[[ZERO]], %[[ACC2:.*]] = %[[ZERO]], %[[ACC3:.*]] = %[[ZERO]]) -> (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) {
// CHECK:         vector.mask {{.*}} { vector.reduction <add>, {{.*}}, {{.*}} fastmath<reassoc> : vector<[4]xf32> into f32 }
// CHECK:         %[[IN_BOUNDS1:.*]] = arith.cmpi slt, {{.*}}, %c127 : index
// CHECK:         %[[PARTIAL1:.*]] = scf.if %[[IN_BOUNDS1]] -> (vector<1xf32>) {
// CHECK:         vector.mask {{.*}} { vector.reduction <add>, {{.*}}, {{.*}} fastmath<reassoc> : vector<[4]xf32> into f32 }
// CHECK:           scf.yield {{.*}} : vector<1xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[ACC1]] : vector<1xf32>
// CHECK:         }
// CHECK:         %[[IN_BOUNDS2:.*]] = arith.cmpi slt, {{.*}}, %c127 : index
// CHECK:         %[[PARTIAL2:.*]] = scf.if %[[IN_BOUNDS2]] -> (vector<1xf32>) {
// CHECK:         vector.mask {{.*}} { vector.reduction <add>, {{.*}}, {{.*}} fastmath<reassoc> : vector<[4]xf32> into f32 }
// CHECK:           scf.yield {{.*}} : vector<1xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[ACC2]] : vector<1xf32>
// CHECK:         }
// CHECK:         %[[IN_BOUNDS3:.*]] = arith.cmpi slt, {{.*}}, %c127 : index
// CHECK:         %[[PARTIAL3:.*]] = scf.if %[[IN_BOUNDS3]] -> (vector<1xf32>) {
// CHECK:         vector.mask {{.*}} { vector.reduction <add>, {{.*}}, {{.*}} fastmath<reassoc> : vector<[4]xf32> into f32 }
// CHECK:           scf.yield {{.*}} : vector<1xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[ACC3]] : vector<1xf32>
// CHECK:         }
// CHECK:         scf.yield
// CHECK:       }
// CHECK:       %[[COMBINE0:.*]] = arith.addf %[[SCHEDULED]]#0, %[[SCHEDULED]]#1 fastmath<reassoc> : vector<1xf32>
// CHECK:       %[[COMBINE1:.*]] = arith.addf %[[COMBINE0]], %[[SCHEDULED]]#2 fastmath<reassoc> : vector<1xf32>
// CHECK:       %[[COMBINE2:.*]] = arith.addf %[[COMBINE1]], %[[SCHEDULED]]#3 fastmath<reassoc> : vector<1xf32>
// CHECK:       return %[[COMBINE2]] : vector<1xf32>

// The pass is intentionally SVE-only: NEON retains the original single
// loop-carried accumulator and reduction association.
// CHECK-NEON-LABEL: func.func @scheduled_masked_add_reduction
// CHECK-NEON:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<1xf32>) {
// CHECK-NEON:         vector.mask {{.*}} { vector.reduction <add>, {{.*}}, {{.*}} : vector<[4]xf32> into f32 }
// CHECK-NEON-NOT:     fastmath<reassoc>
// CHECK-NEON:         scf.yield
// CHECK-NEON:       }
func.func @scheduled_masked_add_reduction(%input: vector<[4]xf32>,
                                          %mask: vector<[4]xi1>,
                                          %initial: vector<1xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %c127 = arith.constant 127 : index
  %vscale = vector.vscale
  %c4 = arith.constant 4 : index
  %step = arith.muli %vscale, %c4 : index
  %result = scf.for %iv = %c0 to %c127 step %step iter_args(%acc = %initial) -> (vector<1xf32>) {
    %scalar_acc = vector.extract %acc[0] : f32 from vector<1xf32>
    %sum = vector.mask %mask {
      vector.reduction <add>, %input, %scalar_acc : vector<[4]xf32> into f32
    } : vector<[4]xi1> -> f32
    %next_acc = vector.broadcast %sum : f32 to vector<1xf32>
    scf.yield %next_acc : vector<1xf32>
  }
  return %result : vector<1xf32>
}

// A transformed reduction result is a different recurrence. It must remain
// untouched rather than being split into independently initialized sums.
// CHECK-LABEL: func.func @scaled_add_reduction_is_not_scheduled
// CHECK:       scf.for {{.*}} iter_args(%{{.*}} = %arg2) -> (vector<1xf32>) {
// CHECK:         vector.mask {{.*}} { vector.reduction <add>, {{.*}}, {{.*}} : vector<[4]xf32> into f32 }
// CHECK:         arith.mulf {{.*}}, %arg3 : f32
// CHECK:         scf.yield {{.*}} : vector<1xf32>
// CHECK:       }
func.func @scaled_add_reduction_is_not_scheduled(
    %input: vector<[4]xf32>, %mask: vector<[4]xi1>,
    %initial: vector<1xf32>, %scale: f32) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %c127 = arith.constant 127 : index
  %vscale = vector.vscale
  %c4 = arith.constant 4 : index
  %step = arith.muli %vscale, %c4 : index
  %result = scf.for %iv = %c0 to %c127 step %step iter_args(%acc = %initial) -> (vector<1xf32>) {
    %scalar_acc = vector.extract %acc[0] : f32 from vector<1xf32>
    %sum = vector.mask %mask {
      vector.reduction <add>, %input, %scalar_acc : vector<[4]xf32> into f32
    } : vector<[4]xi1> -> f32
    %scaled = arith.mulf %sum, %scale : f32
    %next_acc = vector.broadcast %scaled : f32 to vector<1xf32>
    scf.yield %next_acc : vector<1xf32>
  }
  return %result : vector<1xf32>
}
