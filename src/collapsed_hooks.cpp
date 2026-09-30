#include "collapsed_sampler.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Internal entry points into the collapsed kernel layer, used by the unit
// tests to check it against closed forms and reference densities. Indices
// and allocations are 1-based, as in R.

using namespace Rcpp;

// log m of the cluster built by adding the rows `add_idx` and then removing
// the rows `remove_idx` (exercises both directions of the bookkeeping)
// [[Rcpp::export]]
double collapsed_log_m_cpp(
    const arma::mat& Y_N, const arma::imat& Y_C, const arma::mat& Y_P,
    const List& pp, const IntegerVector& add_idx,
    const IntegerVector& remove_idx
) {
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  collapsed::Prior pr = collapsed::make_prior(Y_C, Y_N.n_cols, Y_P.n_cols, n, pp);
  collapsed::Data d = collapsed::make_data(Y_N, Y_C, Y_P, pr);
  collapsed::Stats st = collapsed::empty_stats(pr);
  for (int i : add_idx) collapsed::update_stats(st, d, i - 1, +1);
  for (int i : remove_idx) collapsed::update_stats(st, d, i - 1, -1);
  std::vector<double> work(std::max(pr.qN * pr.qN, 1));
  return collapsed::log_m(pr, st, work.data());
}

// log p(y_i | Y_idx \ {i}) through the samplers' predictive; if i is in idx
// this takes the leave-one-out (in-cluster) path
// [[Rcpp::export]]
double collapsed_log_pred_cpp(
    const arma::mat& Y_N, const arma::imat& Y_C, const arma::mat& Y_P,
    const List& pp, const IntegerVector& idx, int i
) {
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  collapsed::Prior pr = collapsed::make_prior(Y_C, Y_N.n_cols, Y_P.n_cols, n, pp);
  collapsed::Data d = collapsed::make_data(Y_N, Y_C, Y_P, pr);
  collapsed::Stats st = collapsed::empty_stats(pr);
  bool in_st = false;
  for (int j : idx) {
    collapsed::update_stats(st, d, j - 1, +1);
    if (j == i) in_st = true;
  }
  std::vector<double> work(std::max(pr.qN * pr.qN, 1));
  double LN = collapsed::log_m_gauss(pr, st, nullptr, 0, work.data());
  double LN_new;
  return collapsed::log_pred(pr, d, st, LN, i - 1, in_st, work.data(), &LN_new);
}

// Log collapsed posterior of the allocations z in {1, ..., k}, up to a
// constant (alpha is the Dirichlet concentration, so alpha/k per component)
// [[Rcpp::export]]
double collapsed_log_post_cpp(
    const arma::mat& Y_N, const arma::imat& Y_C, const arma::mat& Y_P,
    const List& pp, const arma::ivec& z, int k, double alpha
) {
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  collapsed::Prior pr = collapsed::make_prior(Y_C, Y_N.n_cols, Y_P.n_cols, n, pp);
  collapsed::Data d = collapsed::make_data(Y_N, Y_C, Y_P, pr);
  return collapsed::log_post(pr, d, z - 1, k, alpha / k);
}

// Run the allocation sampler for n_sweeps sweeps without resynchronising the
// cache, then report its drift from a recomputation from z (test T4)
// [[Rcpp::export]]
List collapsed_cache_check_cpp(
    const arma::mat& Y_N, const arma::imat& Y_C, const arma::mat& Y_P,
    const List& pp, int k, double alpha, std::string sampler, int n_sweeps
) {
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  collapsed::Prior pr = collapsed::make_prior(Y_C, Y_N.n_cols, Y_P.n_cols, n, pp);
  collapsed::Data d = collapsed::make_data(Y_N, Y_C, Y_P, pr);
  bool lifted = sampler == "nonreversible";
  collapsed::AllocSampler smp(pr, d, k, alpha / k, 0.5);
  smp.init(lifted);
  for (int s = 0; s < n_sweeps; s++) {
    if (lifted) {
      smp.sweep_nonreversible();
    } else {
      smp.sweep_marginal();
    }
  }
  List out = smp.cache_error();
  out["z"] = smp.z() + 1;
  return out;
}

// Visit counts of every allocation over n_steps single-point updates (P_NR
// steps, or P_MG site updates in scan order) after burn_steps updates. The
// state index is sum_i (z_i - 1) k^(i - 1), the row order of expand.grid()
// (test T6, tiny n only)
// [[Rcpp::export]]
NumericVector collapsed_state_freq_cpp(
    const arma::mat& Y_N, const arma::imat& Y_C, const arma::mat& Y_P,
    const List& pp, int k, double alpha, std::string sampler, double n_steps,
    double burn_steps
) {
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  double n_states = std::pow((double)k, n);
  if (n_states > 1e7) stop("too many states to enumerate");
  collapsed::Prior pr = collapsed::make_prior(Y_C, Y_N.n_cols, Y_P.n_cols, n, pp);
  collapsed::Data d = collapsed::make_data(Y_N, Y_C, Y_P, pr);
  bool lifted = sampler == "nonreversible";
  collapsed::AllocSampler smp(pr, d, k, alpha / k, 0.5);
  smp.init(lifted);

  NumericVector freq((R_xlen_t)n_states);
  double total = burn_steps + n_steps;
  for (double t = 0; t < total; t++) {
    if (lifted) {
      smp.step_nonreversible();
    } else {
      smp.update_site((long)t % n);
    }
    if (t >= burn_steps) {
      const arma::ivec& z = smp.z();
      long idx = 0;
      for (int i = n - 1; i >= 0; i--) idx = idx * k + z(i);
      freq[idx] += 1.0;
    }
  }
  return freq;
}
