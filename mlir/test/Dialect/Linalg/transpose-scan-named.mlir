// RUN: mlir-opt --linalg-fold-transpose-into-extract-insert-slice-pair %s | FileCheck %s

// CHECK-LABEL: func.func @transpose_scan(
// CHECK-SAME:                            %[[IN:.*]]: tensor<10x5xf32>) -> tensor<10x5xf32> {
// CHECK-NOT: linalg.transpose
// CHECK: %{{.*}} = tensor.extract_slice %[[IN]][0, %[[INDEX:.*]]] [10, 1] [1, 1] : tensor<10x5xf32> to tensor<10x1xf32> 
// CHECK: %{{.*}} = tensor.insert_slice %{{.*}} into %{{.*}}[0, %[[INDEX]]] [10, 1] [1, 1] : tensor<10x1xf32> into tensor<10x5xf32>
// CHECK: %[[TRANSPOSED_0:.*]] = linalg.transpose ins(%{{.*}} : tensor<10x5xf32>) outs(%{{.*}} : tensor<5x10xf32>) permutation = [1, 0]
// CHECK: %[[TRANSPOSED_1:.*]] = linalg.transpose ins(%[[TRANSPOSED_0]] : tensor<5x10xf32>) outs(%{{.*}} : tensor<10x5xf32>) permutation = [1, 0]

func.func @transpose_scan(%arg0: tensor<10x5xf32>) -> tensor<10x5xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c5_i32 = arith.constant 5 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<5x10xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
  %0 = tensor.empty() : tensor<5x10xf32>
  %transposed = linalg.transpose ins(%arg0 : tensor<10x5xf32>) outs(%0 : tensor<5x10xf32>) permutation = [1, 0] 
  %1:2 = scf.for %arg1 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg2 = %cst_0, %arg3 = %cst) -> (tensor<10xf32>, tensor<5x10xf32>)  : i32 {
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.maxsi %2, %c0 : index
    %4 = arith.minsi %3, %c4 : index
    %extracted_slice = tensor.extract_slice %transposed[%4, 0] [1, 10] [1, 1] : tensor<5x10xf32> to tensor<1x10xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x10xf32> into tensor<10xf32>
    %5 = tensor.empty() : tensor<10xf32>
    %6 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%arg2, %collapsed : tensor<10xf32>, tensor<10xf32>) outs(%5 : tensor<10xf32>) -> tensor<10xf32>
    %7 = tensor.empty() : tensor<1x10xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<10xf32>) outs(%7 : tensor<1x10xf32>) dimensions = [0] 
    %inserted_slice = tensor.insert_slice %broadcasted into %arg3[%4, 0] [1, 10] [1, 1] : tensor<1x10xf32> into tensor<5x10xf32>
    scf.yield %6, %inserted_slice : tensor<10xf32>, tensor<5x10xf32>
  }
  %8 = tensor.empty() : tensor<10x5xf32>
  %transposed_1 = linalg.transpose ins(%1#1 : tensor<5x10xf32>) outs(%8 : tensor<10x5xf32>) permutation = [1, 0] 
  return %transposed_1 : tensor<10x5xf32>
}