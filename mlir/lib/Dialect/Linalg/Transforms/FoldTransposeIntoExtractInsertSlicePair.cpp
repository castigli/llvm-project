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
#include "llvm/Support/DebugLog.h"

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

    LDBG() << "FoldTransposeExractInsertSlice::matchAndRewrite\n" 
      << op << "\n";

    // check if the operand is defined by a transpose
    auto transposeOp = op.getSource().getDefiningOp<TransposeOp>();
    if (!transposeOp) {
      LDBG() << "Extract slice source is not defined by transpose\n";
      return failure();
    }

    // find insert slice op from yield
    Operation* parentOp = op->getParentOp();
    auto scfForOp = dyn_cast<scf::ForOp>(parentOp);
    if (!scfForOp) {
      LDBG() << "Parent op is not scf.for\n";
      return failure();
    }

    // update scan init operand
    auto scanInit = scfForOp.getInitArgs()[1];
    auto scanInitOp = scanInit.getDefiningOp();
    rewriter.setInsertionPoint(scanInitOp);
    
    auto shapedTy = dyn_cast<ShapedType>(transposeOp.getInput().getType());
    auto staticShape = shapedTy.getShape();
    auto elementType = shapedTy.getElementType();
    auto newScanInit = rewriter.create<arith::ConstantOp>(scanInitOp->getLoc(), rewriter.getZeroAttr(shapedTy));
    // auto newScanInit = rewriter.create<tensor::EmptyOp>(scanInitOp->getLoc(), staticShape, elementType);
    rewriter.replaceAllUsesWith(scanInit, newScanInit);

    // iter arg type for scan needs to be transposed
    auto secondIterArg = scfForOp.getRegionIterArg(1);
    secondIterArg.setType(newScanInit.getType());
    // update result type of scf.for
    scfForOp.getResult(1).setType(newScanInit.getType());


    // second argument of yield should be result of insert slice
    auto yieldOp = dyn_cast<scf::YieldOp>(*(scfForOp.getBody()->getTerminator()));
    // second argument of yield should be result of insert slice
    auto insertSliceOp = yieldOp.getOperand(1).getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) {
      LDBG() << "Second operand of yield is not defined by insert slice\n";
      return failure();
    }
    // check that the insert slice is symmetric to the extract slice
    if (insertSliceOp.getOffsets() != op.getOffsets() ||
        insertSliceOp.getSizes() != op.getSizes() ||
        insertSliceOp.getStrides() != op.getStrides()) {
      LDBG() << "Insert slice and extract slice not symmetric\n";
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
    RankedTensorType curExractOpTy = 
      dyn_cast<RankedTensorType>(op.getResult().getType());
    unsigned sliceRank = curExractOpTy.getRank();

    Type sliceTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(sliceRank,
      op.getSource().getType(), newStaticSizes);
    LDBG() << "Inferred result type: " << sliceTy << "\n";
    // create new extract slice with updated offsets and sizes
    // with same operand as current op
    rewriter.setInsertionPoint(op);
    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
      op.getLoc(), sliceTy, op.getSource(), newOffsets, newSizes, newStrides,
      newStaticOffsets, newStaticSizes, newStaticStrides);

    // to keep IR looking nice, we update the insertion point to be after
    // the insert slice op
    rewriter.setInsertionPointAfter(insertSliceOp);

    // Get source for new insert slice
    Value newInsertSliceSource;
    Value insertSliceSource = insertSliceOp.getSource();
    auto broadcastOp = insertSliceSource.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      LDBG() << "Insert slice source is not defined by broadcast\n";
      newInsertSliceSource = insertSliceSource;
    }
    else {
      // if the source is defined by a broadcast, we need to update the
      // broadcast dimensions according to the transpose permutation
      LDBG() << "Insert slice source is defined by broadcast\n";
      // get dimensions for new broadcast op
      SmallVector<int64_t, 4> broadcastDims;
      for (auto dim : broadcastOp.getDimensions()) {
        broadcastDims.push_back(permutation[dim]);
      }
      auto newBroadcastInitOp = rewriter.create<tensor::EmptyOp>(
        broadcastOp.getLoc(), newExtractOp.getType().getShape(),
        newExtractOp.getType().getElementType());
      LDBG() << "Created new empty op for broadcast init: " 
        << newBroadcastInitOp << "\n";
      // create new broadcast op with updated dimensions
      auto newBroadcastOp = rewriter.create<BroadcastOp>(
        broadcastOp.getLoc(), broadcastOp.getInput(), newBroadcastInitOp, broadcastDims);
      LDBG() << "Created new broadcast op: " << newBroadcastOp << "\n";
      newInsertSliceSource = newBroadcastOp.getResult()[0];
    }

    // create new insert slice with updated offsets and sizes
    auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
      insertSliceOp.getLoc(), newInsertSliceSource, insertSliceOp.getDest(), newOffsets, 
      newSizes, newStrides, newStaticOffsets, newStaticSizes, newStaticStrides);
    LDBG() << "Created new insert slice op: " << newInsertOp << "\n";
    // replace uses of insert slice with new insert slice
    rewriter.replaceAllUsesWith(insertSliceOp, newInsertOp);

    // move transpose after scan (i.e. scf.for)
    // replaces uses of transpose with the input of transpose
    rewriter.replaceAllUsesWith(transposeOp->getOpResult(0), transposeOp.getInput());
    // insert new transpose after the scf.for
    rewriter.setInsertionPointAfter(parentOp);
    Value scanResult = parentOp->getResult(1);
    auto newTransposeOp = rewriter.create<TransposeOp>(parentOp->getLoc(), 
      scanResult, transposeOp.getInit(), transposeOp.getPermutation());
    // replace uses of scan result with the new transpose
    rewriter.replaceAllUsesExcept(scanResult, newTransposeOp->getOpResult(0), newTransposeOp);

    // finally replace extract slice with new extract slice
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
