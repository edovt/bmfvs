#include <Rcpp.h>
#include <cmath>
#include <vector>

using namespace Rcpp;

// Single-pass kernels for the empirical Bayes fixed point (2.10) and the
// posterior probabilities (2.7)-(2.8), replacing bf_weights() on flattened
// S * p vectors. They perform, element by element, exactly the floating-point
// operations of the R reference
//
//   t = (lk0, lk1 + b1, lk2 + b2, lk3 + (b1 + b2)),  m = max(t),
//   e = exp(t - m),  denom = ((e1 + e2) + e3) + e4,  w = e / denom,
//
// and accumulate in long double in storage order, as R's sum() and
// colMeans() do, so the results are bitwise identical to the R code while
// allocating no S * p temporaries.

namespace {

struct Weights {
  double e[4];
  double denom;
  double d0;  // t1 - m, the log of e[0] before exponentiation
  double t[4];  // the unnormalized log terms t1, ..., t4
  double m;     // max(t)
};

inline void bf_terms(double b1, double b2, const double* lk, Weights& out) {
  const double t1 = lk[0];
  const double t2 = lk[1] + b1;
  const double t3 = lk[2] + b2;
  const double b12 = b1 + b2;
  const double t4 = lk[3] + b12;
  double m = t1;               // pmax.int(t1, t2, t3, t4) for finite input
  if (t2 > m) m = t2;
  if (t3 > m) m = t3;
  if (t4 > m) m = t4;
  out.t[0] = t1; out.t[1] = t2; out.t[2] = t3; out.t[3] = t4;
  out.m = m;
  out.d0 = t1 - m;
  out.e[0] = std::exp(out.d0);
  out.e[1] = std::exp(t2 - m);
  out.e[2] = std::exp(t3 - m);
  out.e[3] = std::exp(t4 - m);
  out.denom = out.e[0] + out.e[1] + out.e[2] + out.e[3];
}

}  // namespace

// Sums over all S * p (sample, predictor) pairs of the four normalized
// weights, i.e. N * Phi(kappa) with N = S * p. Same values as
// c(sum(w0), sum(w1), sum(w2), sum(w3)) with w = bf_weights(...).
// [[Rcpp::export]]
NumericVector eb_phi_sums_cpp(NumericVector log_bf1, NumericVector log_bf2,
                              NumericVector log_kappa) {
  const R_xlen_t N = log_bf1.size();
  if (log_bf2.size() != N) stop("log_bf1 and log_bf2 must have the same length");
  if (log_kappa.size() != 4) stop("log_kappa must have length 4");
  const double* b1 = REAL(log_bf1);
  const double* b2 = REAL(log_bf2);
  const double lk[4] = {log_kappa[0], log_kappa[1], log_kappa[2], log_kappa[3]};
  long double s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L;
  Weights w;
  for (R_xlen_t i = 0; i < N; i++) {
    bf_terms(b1[i], b2[i], lk, w);
    s0 += w.e[0] / w.denom;
    s1 += w.e[1] / w.denom;
    s2 += w.e[2] / w.denom;
    s3 += w.e[3] / w.denom;
  }
  return NumericVector::create((double)s0, (double)s1, (double)s2, (double)s3);
}

// Posterior probabilities of (H0j, H1j1, H1j2, H1j3), the Monte Carlo
// averages over the S rows of the S x p log Bayes factor matrices.
//   prob_null: same values as colMeans(matrix(w0, nrow = S)) (long double
//              column sums divided by S, as in R's colMeans());
//   prob_h1:   p x 3 matrix, the same averages of w1, w2, w3;
//   log_prob_null: log of prob_null computed by log-sum-exp of
//              log w0 = (t1 - m) - log(denom), so it stays finite when the
//              average underflows to 0.
//   log_odds_h1: log P(H1j) - log P(H0j) = logit(1 - prob_null), with
//              log P(H1j) the log-mean-exp over samples of
//              log(1 - w0) = lse(t2, t3, t4) - lse(t1, ..., t4), the first
//              log-sum-exp taken about its own maximum. It stays finite and
//              ordered when prob_null rounds to exactly 1 (empirical Bayes at
//              the kappa = (1, 0, 0, 0) corner) or to 0.
// [[Rcpp::export]]
List posterior_probs_cpp(NumericMatrix log_bf1, NumericMatrix log_bf2,
                         NumericVector log_kappa) {
  const int S = log_bf1.nrow(), p = log_bf1.ncol();
  if (log_bf2.nrow() != S || log_bf2.ncol() != p) {
    stop("log_bf1 and log_bf2 must have the same dimensions");
  }
  if (log_kappa.size() != 4) stop("log_kappa must have length 4");
  const double* B1 = REAL(log_bf1);
  const double* B2 = REAL(log_bf2);
  const double lk[4] = {log_kappa[0], log_kappa[1], log_kappa[2], log_kappa[3]};

  NumericVector prob_null(p), log_prob_null(p), log_odds_h1(p);
  NumericMatrix prob_h1(p, 3);
  std::vector<double> lw0(S), lw1(S);
  const long double LS = (long double)S;
  const double log_S = std::log((double)S);
  Weights w;
  for (int j = 0; j < p; j++) {
    const double* b1 = B1 + (size_t)S * j;
    const double* b2 = B2 + (size_t)S * j;
    long double s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L;
    double mx = -INFINITY, mx1 = -INFINITY;
    for (int s = 0; s < S; s++) {
      bf_terms(b1[s], b2[s], lk, w);
      s0 += w.e[0] / w.denom;
      s1 += w.e[1] / w.denom;
      s2 += w.e[2] / w.denom;
      s3 += w.e[3] / w.denom;
      // log w0 = (t1 - m) - log(denom), finite even when e0 underflows
      const double l = w.d0 - std::log(w.denom);
      lw0[s] = l;
      if (l > mx) mx = l;
      // log(1 - w0) = log(e1 + e2 + e3) - log(denom), with the alternative
      // terms rescaled by their own maximum so they cannot underflow
      double ma = w.t[1];
      if (w.t[2] > ma) ma = w.t[2];
      if (w.t[3] > ma) ma = w.t[3];
      const double l1 = (ma - w.m) +
        std::log(std::exp(w.t[1] - ma) + std::exp(w.t[2] - ma) +
                 std::exp(w.t[3] - ma)) -
        std::log(w.denom);
      lw1[s] = l1;
      if (l1 > mx1) mx1 = l1;
    }
    s0 /= LS; s1 /= LS; s2 /= LS; s3 /= LS;
    prob_null[j] = (double)s0;
    prob_h1(j, 0) = (double)s1;
    prob_h1(j, 1) = (double)s2;
    prob_h1(j, 2) = (double)s3;
    if (std::isfinite(mx)) {
      long double acc = 0.0L;
      for (int s = 0; s < S; s++) acc += std::exp(lw0[s] - mx);
      log_prob_null[j] = mx + std::log((double)acc) - log_S;
    } else {
      log_prob_null[j] = mx;  // -Inf only if every log w0 is -Inf
    }
    double log_prob_h1 = mx1;
    if (std::isfinite(mx1)) {
      long double acc = 0.0L;
      for (int s = 0; s < S; s++) acc += std::exp(lw1[s] - mx1);
      log_prob_h1 = mx1 + std::log((double)acc) - log_S;
    }
    log_odds_h1[j] = log_prob_h1 - log_prob_null[j];
  }
  return List::create(Named("prob_null") = prob_null,
                      Named("prob_h1") = prob_h1,
                      Named("log_prob_null") = log_prob_null,
                      Named("log_odds_h1") = log_odds_h1);
}

// log of the mean, over all S * p (sample, predictor) pairs, of BF_j1, BF_j2
// and BF_j3 = BF_j1 * BF_j2, by two-pass log-sum-exp. Near the kappa =
// (1, 0, 0, 0) corner Phi_t(kappa) ~ kappa_t * mean(BF_jt), so the corner
// attracts the empirical Bayes iteration exactly when all three are < 0.
// [[Rcpp::export]]
NumericVector log_mean_bf_cpp(NumericVector log_bf1, NumericVector log_bf2) {
  const R_xlen_t N = log_bf1.size();
  if (log_bf2.size() != N) stop("log_bf1 and log_bf2 must have the same length");
  const double* b1 = REAL(log_bf1);
  const double* b2 = REAL(log_bf2);
  double m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY;
  for (R_xlen_t i = 0; i < N; i++) {
    if (b1[i] > m1) m1 = b1[i];
    if (b2[i] > m2) m2 = b2[i];
    const double b12 = b1[i] + b2[i];
    if (b12 > m3) m3 = b12;
  }
  long double s1 = 0.0L, s2 = 0.0L, s3 = 0.0L;
  for (R_xlen_t i = 0; i < N; i++) {
    s1 += std::exp(b1[i] - m1);
    s2 += std::exp(b2[i] - m2);
    s3 += std::exp(b1[i] + b2[i] - m3);
  }
  const double lN = std::log((double)N);
  return NumericVector::create(
    Named("H1_weights") = m1 + std::log((double)s1) - lN,
    Named("H1_kernel") = m2 + std::log((double)s2) - lN,
    Named("H1_both") = m3 + std::log((double)s3) - lN);
}
