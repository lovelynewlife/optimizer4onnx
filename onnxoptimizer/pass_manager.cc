/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnxoptimizer/pass_manager.h"

namespace ONNX_NAMESPACE {
namespace optimization {

PassManager::PassManager() {}
PassManager::~PassManager() {}

GeneralPassManager::~GeneralPassManager() {
  this->passes.clear();
}
void GeneralPassManager::add(std::shared_ptr<Pass> pass) {
  this->passes.push_back(std::move(pass));
}

std::shared_ptr<PassManagerAnalysis> GeneralPassManager::run(Graph& graph) {
  for (const std::shared_ptr<Pass>& pass : this->passes) {
    auto pass_analysis = pass->runPass(graph);
  }
  return std::shared_ptr<PassManagerAnalysis>(new EmptyPassManagerAnalysis());
}

std::shared_ptr<PassManagerAnalysis> FixedPointPassManager::run(Graph& graph) {
  bool is_graph_changed;
  int max_iters = 100;

  do {
    is_graph_changed = false;
    for (const std::shared_ptr<Pass>& pass : this->passes) {
      std::shared_ptr<PostPassAnalysis> analysis = pass->runPass(graph);
      if (pass->getPassAnalysisType() == PassAnalysisType::Empty) {
        continue;
      }
      std::shared_ptr<CountBasedPassAnalysis> count_analysis =
          std::static_pointer_cast<CountBasedPassAnalysis>(analysis);

      while (count_analysis->fixedPointOptimizationNeeded()) {
        count_analysis = std::static_pointer_cast<CountBasedPassAnalysis>(
            pass->runPass(graph));
        is_graph_changed = is_graph_changed||count_analysis->graphChanged();
      }
      is_graph_changed = is_graph_changed||count_analysis->graphChanged();
    }
    max_iters--;
  } while (is_graph_changed && max_iters > 0);

  return std::shared_ptr<PassManagerAnalysis>(new EmptyPassManagerAnalysis());
}
} // namespace optimization
} // namespace ONNX_NAMESPACE
