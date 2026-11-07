#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

// Bayes factors of Proposition 2.1 for a single posterior sample F.
//
// The Bayes factors are written so that every empty cell contributes exactly
// zero.  Each per-level factor is paired with the prior term it cancels
// against, e.g. for the mixture weights
//
//   log BF_{j1} = -sum_h n_h log w_h
//                 + sum_ell [ sum_h ( lgamma(n_jh(ell) + tau_w w_h)
//                                     - lgamma(tau_w w_h) )
//                             - ( lgamma(n_j(ell) + tau_w) - lgamma(tau_w) ) ],
//
// which removes the separate `d_j * prior_per_level` term and the masking of
// inactive levels, and lets the loops skip empty cells altogether.
//
// All the counting is done in a single pass over observations per predictor,
// shared by the four kernels, and the lgamma() values are read from lookup
// tables indexed by the (integer) cell counts.

static inline double lmvgamma_scalar(double a, int q) {
  double out = q * (q - 1) / 4.0 * std::log(M_PI);
  for (int t = 1; t <= q; t++) out += std::lgamma(a + (1.0 - t) / 2.0);
  return out;
}

// [[Rcpp::export]]
List log_bf_sample_cpp(
    IntegerVector z_,          // 0-based cluster labels, length n
    IntegerMatrix X_,          // n x p predictors, values 0, ..., d_j - 1
    IntegerVector d_j_,        // number of levels per predictor
    int k,
    NumericVector w_,          // mixture weights, length k
    double tau_omega,
    NumericMatrix Y_N_,        // n x qN continuous responses (0 cols if absent)
    List mu_,                  // k vectors of length qN
    List Sigma_,               // k matrices qN x qN
    double tau_mu,
    double tau_sigma,
    IntegerMatrix Y_C_,        // n x qC categorical responses, 0-based
    List psi_,                 // qC matrices k x m_t
    double tau_psi,
    NumericMatrix Y_P_,        // n x qP count responses
    NumericMatrix rho_,        // k x qP
    double tau_rho,
    int n_threads
) {
  const int n = z_.size();
  const int p = X_.ncol();
  const int qN = Y_N_.ncol();
  const int qC = Y_C_.ncol();
  const int qP = Y_P_.ncol();

  const int *X = INTEGER(X_);
  const int *z = INTEGER(z_);
  const int *dj = INTEGER(d_j_);
  const int *YC = qC > 0 ? INTEGER(Y_C_) : nullptr;
  const double *YP = qP > 0 ? REAL(Y_P_) : nullptr;
  const double *YN = qN > 0 ? REAL(Y_N_) : nullptr;

  int max_d = 1;
  for (int j = 0; j < p; j++) if (dj[j] > max_d) max_d = dj[j];

  std::vector<int> n_h(k, 0);
  for (int i = 0; i < n; i++) n_h[z[i]]++;

  // ---------------------------------------------------------------- BF_{j1}
  // LGW[h][c] = lgamma(c + tau_w w_h) - lgamma(tau_w w_h)
  // LGO[c]    = lgamma(c + tau_w)     - lgamma(tau_w)
  std::vector<double> LGW((size_t)k * (n + 1)), LGO(n + 1);
  double bf1_shared = 0.0;
  {
    const double *w = REAL(w_);
    for (int h = 0; h < k; h++) {
      const double a = tau_omega * w[h], l0 = std::lgamma(a);
      double *row = &LGW[(size_t)h * (n + 1)];
      for (int c = 0; c <= n; c++) row[c] = std::lgamma(c + a) - l0;
      bf1_shared -= n_h[h] * std::log(w[h]);
    }
    const double l0 = std::lgamma(tau_omega);
    for (int c = 0; c <= n; c++) LGO[c] = std::lgamma(c + tau_omega) - l0;
  }

  // ------------------------------------------------- BF_{j2} product multinom
  std::vector<int> m(qC), msum(qC + 1, 0);
  for (int t = 0; t < qC; t++) {
    NumericMatrix P = psi_[t];
    m[t] = P.ncol();
    msum[t + 1] = msum[t] + m[t];
  }
  const int mtot = qC > 0 ? msum[qC] : 0;
  const int cblock = k * mtot;                       // cells per level
  std::vector<double> DC((size_t)cblock * (n + 1)), DCtau(n + 1);
  double bf2c_shared = 0.0;
  if (qC > 0) {
    // Flatten psi first (R API is not thread safe), then fill the table in
    // parallel: it has k * sum_t m_t * (n + 1) entries and is rebuilt for
    // every posterior sample.
    std::vector<double> logpsi(cblock), alpha(cblock);
    for (int t = 0; t < qC; t++) {
      NumericMatrix P = psi_[t];
      for (int h = 0; h < k; h++)
        for (int s = 0; s < m[t]; s++) {
          const int idx = h * mtot + msum[t] + s;
          alpha[idx] = tau_psi * P(h, s);
          logpsi[idx] = std::log(P(h, s));
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_threads > 1) num_threads(n_threads)
#endif
    for (int idx = 0; idx < cblock; idx++) {
      const double a = alpha[idx], l0 = std::lgamma(a);
      double *row = &DC[(size_t)idx * (n + 1)];
      for (int c = 0; c <= n; c++) row[c] = std::lgamma(c + a) - l0;
    }
    const double l0 = std::lgamma(tau_psi);
    for (int c = 0; c <= n; c++) DCtau[c] = std::lgamma(c + tau_psi) - l0;
    // sum_{h,t,s} n_hts log psi_hts
    std::vector<int> n_hts(cblock, 0);
    for (int i = 0; i < n; i++)
      for (int t = 0; t < qC; t++)
        n_hts[z[i] * mtot + msum[t] + YC[i + (size_t)n * t]]++;
    for (int b = 0; b < cblock; b++) bf2c_shared += n_hts[b] * logpsi[b];
  }

  // --------------------------------------------------- BF_{j2} product Poisson
  std::vector<double> pois_a(k * std::max(qP, 1)), pois_lga(k * std::max(qP, 1));
  std::vector<double> LTR(qP > 0 ? n + 1 : 0);  // log(tau_rho + c)
  double bf2p_shared = 0.0, log_tau_rho = 0.0;
  if (qP > 0) {
    log_tau_rho = std::log(tau_rho);
    for (int c = 0; c <= n; c++) LTR[c] = std::log(tau_rho + c);
    for (int h = 0; h < k; h++)
      for (int t = 0; t < qP; t++) {
        const double r = rho_(h, t);
        pois_a[h * qP + t] = tau_rho * r;
        pois_lga[h * qP + t] = std::lgamma(tau_rho * r);
        bf2p_shared += n_h[h] * r;
      }
    for (int i = 0; i < n; i++)
      for (int t = 0; t < qP; t++)
        bf2p_shared -= YP[i + (size_t)n * t] * std::log(rho_(z[i], t));
  }

  // -------------------------------------------------------- BF_{j2} Gaussian
  const int qN2 = qN * (qN + 1) / 2;
  const int kN = qN > 0 ? k : 0;
  std::vector<arma::mat> Sig_inv(kN), tau_Sig(kN);
  std::vector<arma::vec> mu_h(kN);
  std::vector<double> P_h(kN), LMG(qN > 0 ? n + 1 : 0), LTM(qN > 0 ? n + 1 : 0);
  double bf2n_shared = 0.0;
  if (qN > 0) {
    const double lg_prior = lmvgamma_scalar(tau_sigma / 2.0, qN);
    for (int h = 0; h < k; h++) {
      arma::mat S = as<arma::mat>(Sigma_[h]);
      arma::vec mv = as<arma::vec>(mu_[h]);
      double sign, ld;
      arma::log_det(ld, sign, S);
      Sig_inv[h] = arma::inv_sympd(S);
      tau_Sig[h] = tau_sigma * S;
      mu_h[h] = mv;
      // log|tau_sigma * Sigma_h| = qN log(tau_sigma) + log|Sigma_h|
      P_h[h] = tau_sigma / 2.0 * (qN * std::log(tau_sigma) + ld) +
               qN / 2.0 * std::log(tau_mu) - lg_prior;
      bf2n_shared += n_h[h] / 2.0 * ld;
    }
    for (int i = 0; i < n; i++) {           // 1/2 sum_i (y_i - mu)' Sinv (y_i - mu)
      const int h = z[i];
      double quad = 0.0;
      for (int a = 0; a < qN; a++) {
        const double da = YN[i + (size_t)n * a] - mu_h[h][a];
        for (int b = 0; b < qN; b++)
          quad += da * Sig_inv[h](a, b) * (YN[i + (size_t)n * b] - mu_h[h][b]);
      }
      bf2n_shared += quad / 2.0;
    }
    for (int c = 0; c <= n; c++) LMG[c] = lmvgamma_scalar((tau_sigma + c) / 2.0, qN);
    // log(tau_mu + c), formerly evaluated once per non-empty cell
    for (int c = 0; c <= n; c++) LTM[c] = std::log(tau_mu + c);
  }

  // ------------------------------------------------------ per-predictor pass
  NumericVector out1(p), out2(p);
  double *o1 = REAL(out1), *o2 = REAL(out2);
  const bool gauss1_only = (qN == 1 && qC == 0 && qP == 0);

#ifdef _OPENMP
#pragma omp parallel if (n_threads > 1) num_threads(n_threads)
#endif
  {
    std::vector<int> cnt((size_t)max_d * k), cnt_lev(max_d);
    std::vector<int> ccnt((size_t)max_d * std::max(cblock, 1));
    std::vector<double> psum((size_t)max_d * k * std::max(qP, 1));
    std::vector<double> gsum((size_t)max_d * k * std::max(qN, 1));
    std::vector<double> gss((size_t)max_d * k * std::max(qN2, 1));
    arma::mat Lam(std::max(qN, 1), std::max(qN, 1)), R(std::max(qN, 1), std::max(qN, 1));
    std::vector<double> ybar(std::max(qN, 1)), dev(std::max(qN, 1));

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int j = 0; j < p; j++) {
      const int d = dj[j];
      const int *Xj = X + (size_t)n * j;

      std::fill_n(cnt.begin(), (size_t)d * k, 0);
      std::fill_n(cnt_lev.begin(), d, 0);
      if (qC > 0) std::fill_n(ccnt.begin(), (size_t)d * cblock, 0);
      if (qP > 0) std::fill_n(psum.begin(), (size_t)d * k * qP, 0.0);
      if (qN > 0) {
        std::fill_n(gsum.begin(), (size_t)d * k * qN, 0.0);
        std::fill_n(gss.begin(), (size_t)d * k * qN2, 0.0);
      }

      // ---- one pass over observations: fill every cell statistic ----------
      // Pure univariate Gaussian response (the common screening case): the
      // same accumulations in the same order, without the per-observation
      // block dispatch; n_j(ell) is then the (exact, integer) sum of the
      // n_jh(ell) over h.
      if (gauss1_only) {
        for (int i = 0; i < n; i++) {
          const size_t cell = (size_t)Xj[i] * k + z[i];
          const double ya = YN[i];
          cnt[cell]++;
          gsum[cell] += ya;
          gss[cell] += ya * ya;
        }
        for (int ell = 0; ell < d; ell++) {
          int tot = 0;
          for (int h = 0; h < k; h++) tot += cnt[(size_t)ell * k + h];
          cnt_lev[ell] = tot;
        }
      } else
      for (int i = 0; i < n; i++) {
        const int ell = Xj[i], h = z[i];
        const size_t cell = (size_t)ell * k + h;
        cnt[cell]++;
        cnt_lev[ell]++;
        if (qC > 0) {
          int *base = &ccnt[(size_t)ell * cblock + (size_t)h * mtot];
          for (int t = 0; t < qC; t++) base[msum[t] + YC[i + (size_t)n * t]]++;
        }
        if (qP > 0) {
          double *ps = &psum[cell * qP];
          for (int t = 0; t < qP; t++) ps[t] += YP[i + (size_t)n * t];
        }
        if (qN > 0) {
          double *gs = &gsum[cell * qN];
          double *gq = &gss[cell * qN2];
          int u = 0;
          for (int a = 0; a < qN; a++) {
            const double ya = YN[i + (size_t)n * a];
            gs[a] += ya;
            for (int b = a; b < qN; b++) gq[u++] += ya * YN[i + (size_t)n * b];
          }
        }
      }

      // ---- accumulate; empty cells contribute exactly zero ----------------
      double v1 = bf1_shared, v2 = 0.0;
      for (int ell = 0; ell < d; ell++) {
        if (cnt_lev[ell] == 0) continue;             // whole level absent
        v1 -= LGO[cnt_lev[ell]];
        if (qC > 0) {
          const int *cl = &ccnt[(size_t)ell * cblock];
          for (int b = 0; b < cblock; b++)
            if (cl[b]) v2 += DC[(size_t)b * (n + 1) + cl[b]];
        }
        for (int h = 0; h < k; h++) {
          const int c = cnt[(size_t)ell * k + h];
          if (c == 0) continue;
          v1 += LGW[(size_t)h * (n + 1) + c];
          if (qC > 0) v2 -= qC * DCtau[c];
          if (qP > 0) {
            const double *ps = &psum[((size_t)ell * k + h) * qP];
            const double ltn = LTR[c];
            for (int t = 0; t < qP; t++) {
              const double a = pois_a[h * qP + t];
              v2 += std::lgamma(a + ps[t]) - pois_lga[h * qP + t] +
                    a * log_tau_rho - (a + ps[t]) * ltn;
            }
          }
          if (qN > 0) {
            const double *gs = &gsum[((size_t)ell * k + h) * qN];
            const double *gq = &gss[((size_t)ell * k + h) * qN2];
            for (int a = 0; a < qN; a++) {
              ybar[a] = gs[a] / c;
              dev[a] = ybar[a] - mu_h[h][a];
            }
            const double kf = tau_mu * c / (tau_mu + c);
            int u = 0;
            for (int a = 0; a < qN; a++)
              for (int b = a; b < qN; b++) {
                const double val = tau_Sig[h](a, b) + gq[u++] -
                                   c * ybar[a] * ybar[b] + kf * dev[a] * dev[b];
                Lam(a, b) = val;
                Lam(b, a) = val;
              }
            double ld_Lam;
            if (qN == 1 && Lam(0, 0) > 0.0) {
              // 1 x 1 Cholesky: R = sqrt(Lambda), and the general branch's
              // 0.0 + 2 log R(0, 0); the same operations as LAPACK potrf
              // followed by the loop below, without the LAPACK call
              ld_Lam = 2.0 * std::log(std::sqrt(Lam(0, 0)));
            } else if (arma::chol(R, Lam)) {
              ld_Lam = 0.0;
              for (int a = 0; a < qN; a++) ld_Lam += 2.0 * std::log(R(a, a));
            } else {
              double sign;
              arma::log_det(ld_Lam, sign, Lam);
            }
            v2 += c * qN / 2.0 * M_LN2 + LMG[c] -
                  (tau_sigma + c) / 2.0 * ld_Lam -
                  qN / 2.0 * LTM[c] + P_h[h];
          }
        }
      }
      o1[j] = v1;
      o2[j] = v2 - bf2c_shared + bf2p_shared + bf2n_shared;
    }
  }

  return List::create(Named("log_bf1") = out1, Named("log_bf2") = out2);
}
