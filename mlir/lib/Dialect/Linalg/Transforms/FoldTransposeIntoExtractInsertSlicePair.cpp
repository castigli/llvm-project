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
    if (!transposeOp)
      return failure();

    // // check if user is unique and is a collapse shape op
    // auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(*op.getResult().getUsers().begin());
    // if (!collapseShapeOp || !collapseShapeOp->hasOneUse())
    //   return failure();

    // // check that there are only elementwise ops between collapse shape and yield
    // Operation* currentOp = *collapseShapeOp.getResult().getUsers().begin();
    // Operation* broadcastOp = nullptr;
    // Operation* insertSliceOp = nullptr;
    // while(dyn_cast<scf::YieldOp>(currentOp) == nullptr) {
    //   llvm::dbgs() << "Elementwise check inspecting op: " << *currentOp << "\n";
    //   if (!isa<ElementwiseOp>(currentOp)) {
    //     llvm::dbgs() << "Elementwise check failed!\n";
    //     return failure();
    //   }
    //   // TODO check all the users
    //   currentOp = *currentOp->getResult(0).getUsers().begin();
    // }


    // check that collapse shape and broadcast are inverses
    auto newExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
      op.getLoc(), op.getResult().getType(), op.getSource(), op.getOffsets(), op.getSizes(), op.getStrides(), op.getStaticOffsets(), op.getStaticSizes(), op.getStaticStrides());

    // move transpose after scf.for
    // replaces uses of transpose with the input of transpose
    Value transposeInput = transposeOp.getInput();
    Value transposeInit = transposeOp.getInit();
    Value transposeOutput = transposeOp->getOpResult(0);
    rewriter.replaceAllUsesWith(transposeOutput, transposeInput);
    // insert new transpose after the scf.for
    Operation* parentOp = op->getParentOp();
    rewriter.setInsertionPointAfter(parentOp);
    Value scanResult = parentOp->getResult(1);
    auto newTransposeOp = rewriter.create<TransposeOp>(parentOp->getLoc(), 
      scanResult, transposeInit, transposeOp.getPermutation());
    // replace uses of scan result with the new transpose
    rewriter.replaceAllUsesExcept(scanResult, newTransposeOp->getOpResult(0), newTransposeOp);

    rewriter.replaceOp(op, newExtractSlice);
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
