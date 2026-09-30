#ifndef BMFVS_COLLAPSED_H
#define BMFVS_COLLAPSED_H

#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

// Collapsed representation of the mixed-kernel mixture (Supplemental C.4):
// the component parameters theta_h are integrated out, so each cluster is
// summarised by its conjugate sufficient statistics and scored by its log
// marginal likelihood log m(Y_h). Shared by the marginal and non-reversible
// allocation samplers and by the unit-test entry points.
//
// Nothing is dropped from log m (the log(y!) terms of the count block are
// tracked as a sufficient statistic), so every quantity here is exact and
// log m(empty) = 0.
namespace collapsed {

// Observations rearranged so that subject i is a contiguous column
struct Data {
  int n, qN, qC, qP;
  arma::mat YN;    // qN x n, continuous responses
  arma::imat YC;   // qC x n, flat level index off(t) + y_it - 1
  arma::mat YP;    // qP x n, count responses
  arma::vec lfact; // n, sum_t log(y_it!) over the count block
};

// Prior hyperparameters and precomputed constants
struct Prior {
  int qN, qC, qP, n_levels;
  // Gaussian: mu | Sigma ~ N(0, Sigma / beta0), Sigma ~ IW(nu0, eps0 I)
  double beta0, nu0, eps0;
  std::vector<double> lmvg; // log Gamma_q((nu0 + n) / 2), n = 0, 1, ...
  double gauss_const;       // nu0/2 q log(eps0) - log Gamma_q(nu0/2) + q/2 log(beta0)
  // Categorical: psi_t ~ Dir(alpha_c[t] / m_t, ...)
  arma::ivec m, off;        // levels per response, offsets into flat counts
  arma::vec alpha_c;
  // Poisson: rho_t ~ Ga(alpha_p[t], beta_p[t]), shape-rate
  arma::vec alpha_p, beta_p;
};

// Sufficient statistics of one cluster
struct Stats {
  int n = 0;
  arma::vec s;         // qN, sum of y
  arma::mat Q;         // qN x qN, sum of y y^T (lower triangle only)
  arma::ivec cnt;      // n_levels, level counts (flat over responses)
  arma::vec S;         // qP, sum of counts
  double lfact = 0.0;  // sum of log(y!) over members
};

// Read a numeric prior entry from the list built by expand_prior_params()
inline arma::vec prior_entry(const Rcpp::List& pp, const char* name) {
  if (!pp.containsElementNamed(name) || Rf_isNull(pp[name])) {
    return arma::vec();
  }
  return Rcpp::as<arma::vec>(pp[name]);
}

// n_max bounds the cluster sizes that will be scored (the sample size)
inline Prior make_prior(
    const arma::imat& Y_C, int qN, int qP, int n_max, const Rcpp::List& pp
) {
  Prior pr;
  pr.qN = qN;
  pr.qC = Y_C.n_cols;
  pr.qP = qP;

  pr.beta0 = prior_entry(pp, "beta0")(0);
  pr.nu0 = prior_entry(pp, "nu0")(0);
  pr.eps0 = prior_entry(pp, "eps0")(0);
  if (qN > 0) {
    // log Gamma_q(x) = q(q-1)/4 log(pi) + sum_j lgamma(x + (1 - j)/2)
    pr.lmvg.resize(n_max + 2);
    for (int n = 0; n < n_max + 2; n++) {
      double x = 0.5 * (pr.nu0 + n);
      double v = 0.25 * qN * (qN - 1) * std::log(M_PI);
      for (int j = 1; j <= qN; j++) v += std::lgamma(x + 0.5 * (1 - j));
      pr.lmvg[n] = v;
    }
    pr.gauss_const = 0.5 * pr.nu0 * qN * std::log(pr.eps0) - pr.lmvg[0] +
                     0.5 * qN * std::log(pr.beta0);
  }

  // Categorical levels are recoded to 1, ..., m_t by split_response()
  pr.m.zeros(pr.qC);
  pr.off.zeros(pr.qC);
  pr.n_levels = 0;
  for (int t = 0; t < pr.qC; t++) {
    pr.m(t) = Y_C.col(t).max();
    pr.off(t) = pr.n_levels;
    pr.n_levels += pr.m(t);
  }
  pr.alpha_c = prior_entry(pp, "alpha_c");
  if (pr.alpha_c.n_elem == 0) pr.alpha_c = arma::conv_to<arma::vec>::from(pr.m);

  pr.alpha_p = prior_entry(pp, "alpha_p");
  pr.beta_p = prior_entry(pp, "beta_p");
  if (qP > 0 && (pr.alpha_p.n_elem < (arma::uword)qP ||
                 pr.beta_p.n_elem < (arma::uword)qP)) {
    Rcpp::stop("alpha_p and beta_p must have one entry per count response");
  }
  if (pr.qC > 0 && pr.alpha_c.n_elem < (arma::uword)pr.qC) {
    Rcpp::stop("alpha_c must have one entry per categorical response");
  }
  return pr;
}

inline Data make_data(
    const arma::mat& Y_N, const arma::imat& Y_C, const arma::mat& Y_P,
    const Prior& pr
) {
  Data d;
  d.qN = Y_N.n_cols;
  d.qC = Y_C.n_cols;
  d.qP = Y_P.n_cols;
  d.n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  d.YN = Y_N.t();
  d.YC.set_size(d.qC, d.n);
  for (int i = 0; i < d.n; i++) {
    for (int t = 0; t < d.qC; t++) d.YC(t, i) = pr.off(t) + Y_C(i, t) - 1;
  }
  d.YP = Y_P.t();
  d.lfact.zeros(d.n);
  for (int i = 0; i < d.n; i++) {
    for (int t = 0; t < d.qP; t++) d.lfact(i) += std::lgamma(Y_P(i, t) + 1.0);
  }
  return d;
}

inline Stats empty_stats(const Prior& pr) {
  Stats st;
  st.s.zeros(pr.qN);
  st.Q.zeros(pr.qN, pr.qN);
  st.cnt.zeros(pr.n_levels);
  st.S.zeros(pr.qP);
  return st;
}

// Add (sign = +1) or remove (sign = -1) subject i
inline void update_stats(Stats& st, const Data& d, int i, int sign) {
  st.n += sign;
  if (d.qN > 0) {
    const double* y = d.YN.colptr(i);
    for (int b = 0; b < d.qN; b++) {
      st.s(b) += sign * y[b];
      for (int a = b; a < d.qN; a++) st.Q(a, b) += sign * y[a] * y[b];
    }
  }
  for (int t = 0; t < d.qC; t++) st.cnt(d.YC(t, i)) += sign;
  if (d.qP > 0) {
    const double* y = d.YP.colptr(i);
    for (int t = 0; t < d.qP; t++) st.S(t) += sign * y[t];
    st.lfact += sign * d.lfact(i);
  }
}

// In-place lower Cholesky of the q x q column-major matrix A (lower triangle
// read), returning log|A|. The q here is small, so a plain loop beats LAPACK.
inline double chol_logdet(double* A, int q) {
  double ld = 0.0;
  for (int j = 0; j < q; j++) {
    double d = A[j + q * j];
    for (int k = 0; k < j; k++) d -= A[j + q * k] * A[j + q * k];
    if (!(d > 0.0)) {
      Rcpp::stop("collapsed sampler: Psi_n is not positive definite");
    }
    double ljj = std::sqrt(d);
    A[j + q * j] = ljj;
    ld += std::log(d);
    for (int i = j + 1; i < q; i++) {
      double v = A[i + q * j];
      for (int k = 0; k < j; k++) v -= A[i + q * k] * A[j + q * k];
      A[i + q * j] = v / ljj;
    }
  }
  return ld;
}

// Gaussian (NIW) log marginal likelihood of the cluster, optionally with the
// point y added (sign = +1) or removed (sign = -1) without touching st.
// Psi_n = eps0 I + Q - s s^T / beta_n; `work` holds qN * qN doubles.
inline double log_m_gauss(
    const Prior& pr, const Stats& st, const double* y, int sign, double* work
) {
  const int q = pr.qN;
  const int n = st.n + (y ? sign : 0);
  if (q == 0 || n == 0) return 0.0;
  const double bn = pr.beta0 + n;
  for (int b = 0; b < q; b++) {
    double sb = st.s(b) + (y ? sign * y[b] : 0.0);
    for (int a = b; a < q; a++) {
      double sa = st.s(a) + (y ? sign * y[a] : 0.0);
      double v = st.Q(a, b) + (y ? sign * y[a] * y[b] : 0.0) - sa * sb / bn;
      if (a == b) v += pr.eps0;
      work[a + q * b] = v;
    }
  }
  double ld = chol_logdet(work, q);
  return pr.gauss_const + pr.lmvg[n] - 0.5 * n * q * std::log(M_PI) -
         0.5 * (pr.nu0 + n) * ld - 0.5 * q * std::log(bn);
}

// Product multinomial (Dirichlet) log marginal likelihood
inline double log_m_cat(const Prior& pr, const Stats& st) {
  if (pr.qC == 0 || st.n == 0) return 0.0;
  double out = 0.0;
  for (int t = 0; t < pr.qC; t++) {
    double a = pr.alpha_c(t), am = a / pr.m(t);
    out += std::lgamma(a) - std::lgamma(a + st.n);
    for (int l = 0; l < pr.m(t); l++) {
      int c = st.cnt(pr.off(t) + l);
      if (c > 0) out += std::lgamma(am + c) - std::lgamma(am);
    }
  }
  return out;
}

// Product Poisson (Gamma) log marginal likelihood
inline double log_m_pois(const Prior& pr, const Stats& st) {
  if (pr.qP == 0 || st.n == 0) return 0.0;
  double out = -st.lfact;
  for (int t = 0; t < pr.qP; t++) {
    double a = pr.alpha_p(t), b = pr.beta_p(t), S = st.S(t);
    out += a * std::log(b) - std::lgamma(a) + std::lgamma(a + S) -
           (a + S) * std::log(b + st.n);
  }
  return out;
}

// Full log m(Y_h): the blocks are independent given the allocation
inline double log_m(const Prior& pr, const Stats& st, double* work) {
  return log_m_gauss(pr, st, nullptr, 0, work) + log_m_cat(pr, st) +
         log_m_pois(pr, st);
}

// Log posterior predictive log p(y_i | D) = log m(D + i) - log m(D), where
//   in_st = false: D is the cluster st (i is not a member);
//   in_st = true:  D is st without i (i is a member, leave-one-out).
// LN is the cached Gaussian log m of st. On return *LN_new holds the
// Gaussian log m of the cluster after moving i in (in_st = false) or out
// (in_st = true), so the sampler can update its cache on acceptance.
// The categorical and Poisson parts are closed-form and cost O(q).
inline double log_pred(
    const Prior& pr, const Data& d, const Stats& st, double LN, int i,
    bool in_st, double* work, double* LN_new
) {
  double out = 0.0;
  const int e = in_st ? 1 : 0;
  const int n = st.n - e;

  if (pr.qN > 0) {
    const double* y = d.YN.colptr(i);
    if (in_st) {
      *LN_new = log_m_gauss(pr, st, y, -1, work);
      out += LN - *LN_new;
    } else {
      *LN_new = log_m_gauss(pr, st, y, +1, work);
      out += *LN_new - LN;
    }
  } else {
    *LN_new = 0.0;
  }

  // prod_t (alpha_t/m_t + n_{t,y_t}) / (alpha_t + n)
  for (int t = 0; t < pr.qC; t++) {
    double a = pr.alpha_c(t);
    out += std::log(a / pr.m(t) + st.cnt(d.YC(t, i)) - e) - std::log(a + n);
  }

  // prod_t NegBin(y_t | size = a + S, prob = (b + n) / (b + n + 1))
  if (pr.qP > 0) {
    const double* y = d.YP.colptr(i);
    for (int t = 0; t < pr.qP; t++) {
      double a = pr.alpha_p(t), b = pr.beta_p(t);
      double aS = a + st.S(t) - e * y[t];
      out += std::lgamma(aS + y[t]) - std::lgamma(aS) + aS * std::log(b + n) -
             (aS + y[t]) * std::log(b + n + 1.0);
    }
    out -= d.lfact(i);
  }
  return out;
}

// Build the per-cluster statistics from 0-based allocations z
inline std::vector<Stats> stats_from_z(
    const Prior& pr, const Data& d, const arma::ivec& z, int k
) {
  std::vector<Stats> st(k, empty_stats(pr));
  for (int i = 0; i < d.n; i++) update_stats(st[z(i)], d, i, +1);
  return st;
}

// Log collapsed posterior of the allocations, up to a constant:
//   sum_h lgamma(alpha/k + n_h) + log m(Y_h)
inline double log_post(
    const Prior& pr, const Data& d, const arma::ivec& z, int k, double a_k
) {
  std::vector<Stats> st = stats_from_z(pr, d, z, k);
  std::vector<double> work(std::max(pr.qN * pr.qN, 1));
  double out = 0.0;
  for (int h = 0; h < k; h++) {
    out += std::lgamma(a_k + st[h].n) + log_m(pr, st[h], work.data());
  }
  return out;
}

} // namespace collapsed

#endif
