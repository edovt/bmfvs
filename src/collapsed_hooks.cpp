#include "collapsed.h"
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

