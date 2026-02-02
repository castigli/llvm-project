//===- BlockPackMatmul.cpp - Linalg matmul block packing ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGFOLDTRANSPOSEINTOEXTRACTINSERTSLICEPAIRPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-fold-transpose-into-extract-insert-slice-pair"

namespace {
struct FoldTransposeExractInsertSlice : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {

    llvm::dbgs() << "FoldTransposeExractInsertSlice::matchAndRewrite\n" 
      << op << "\n";

    // check if the operand is defined by a transpose
    auto transposeOp = op.getSource().getDefiningOp<TransposeOp>();
    if (!transposeOp) {
      llvm::dbgs() << "Extract slice source is not defined by transpose\n";
      return failure();
    }

    // check if user is unique and is a collapse shape op
    auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(*op.getResult().getUsers().begin());
    if (!collapseShapeOp || !collapseShapeOp->hasOneUse())
      return failure();

    // find insert slice op from yield
    Operation* parentOp = op->getParentOp();
    auto scfForOp = dyn_cast<scf::ForOp>(parentOp);
    if (!scfForOp)
      return failure();
    auto yieldOp = dyn_cast<scf::YieldOp>(*(scfForOp.getBody()->getTerminator()));
    // second argument of yield should be result of insert slice
    auto insertSliceOp = yieldOp.getOperand(1).getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp)
      return failure();
    // check that the insert slice is symmetric to the extract slice
    if (insertSliceOp.getOffsets() != op.getOffsets() ||
        insertSliceOp.getSizes() != op.getSizes() ||
        insertSliceOp.getStrides() != op.getStrides()) {
      llvm::dbgs() << "Insert slice and extract slice not symmetric\n";
      return failure();
    }
    // exchange order of offsets and sizes according to transpose permutation
    SmallVector<Value, 4> newOffsets;
    SmallVector<Value, 4> newSizes;
    SmallVector<Value, 4> newStrides;
    SmallVector<int64_t, 4> newStaticOffsets;
    SmallVector<int64_t, 4> newStaticSizes;
    SmallVector<int64_t, 4> newStaticStrides;
    auto permutation = transposeOp.getPermutation();
    
    // build dynamic offsets
    uint32_t iDynOffset = 0;
    SmallVector<Value, 4> dynOffsets;
    for(auto staticOffset : op.getStaticOffsets()) {
      if (staticOffset == ShapedType::kDynamic) {
        dynOffsets.push_back(op.getOffsets()[iDynOffset]);
        iDynOffset++;
      }
      else {
        dynOffsets.push_back(nullptr);
      }
    }

    // get new static and dynamic offsets/sizes/strides
    for (auto perm : permutation) {
      auto staticOffset = op.getStaticOffsets()[perm];
      newStaticOffsets.push_back(staticOffset);
      if (staticOffset == ShapedType::kDynamic)
        newOffsets.push_back(dynOffsets[perm]);
      // TODO allow for dynamic sizes/strides
      newStaticSizes.push_back(op.getStaticSizes()[perm]);
      newStaticStrides.push_back(op.getStaticStrides()[perm]);  
    }

    // get result type of new extract slice
    Type resultType = tensor::ExtractSliceOp::inferResultType(
      op.getSource().getType(), newStaticSizes);
    llvm::dbgs() << "Inferred result type: " << resultType << "\n";
    // create new extract slice with updated offsets and sizes
    // with same operand as current op
    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
      op.getLoc(), resultType, op.getSource(), newOffsets, newSizes, newStrides,
      newStaticOffsets, newStaticSizes, newStaticStrides);

    // TODO replace broadcast and insert slice with new ones

    // move transpose after scf.for
    // replaces uses of transpose with the input of transpose
    Value transposeInput = transposeOp.getInput();
    Value transposeInit = transposeOp.getInit();
    Value transposeOutput = transposeOp->getOpResult(0);
    rewriter.replaceAllUsesWith(transposeOutput, transposeInput);
    // insert new transpose after the scf.for
    rewriter.setInsertionPointAfter(parentOp);
    Value scanResult = parentOp->getResult(1);
    auto newTransposeOp = rewriter.create<TransposeOp>(parentOp->getLoc(), 
      scanResult, transposeInit, transposeOp.getPermutation());
    // replace uses of scan result with the new transpose
    rewriter.replaceAllUsesExcept(scanResult, newTransposeOp->getOpResult(0), newTransposeOp);

    rewriter.replaceOp(op, newExtractOp);
    return success();
  }
};

struct LinalgFoldTransposeIntoExtractInsertSlicePairPass
    : public impl::LinalgFoldTransposeIntoExtractInsertSlicePairPassBase<
          LinalgFoldTransposeIntoExtractInsertSlicePairPass> {
  using impl::LinalgFoldTransposeIntoExtractInsertSlicePairPassBase<
      LinalgFoldTransposeIntoExtractInsertSlicePairPass>::LinalgFoldTransposeIntoExtractInsertSlicePairPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateLinalgFoldTransposeIntoExtractInsertSlicePairPatterns(patterns);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void mlir::linalg::populateLinalgFoldTransposeIntoExtractInsertSlicePairPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldTransposeExractInsertSlice>(patterns.getContext());
}
