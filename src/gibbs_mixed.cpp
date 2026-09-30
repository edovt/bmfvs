#include "mixture.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Unified conditional Gibbs sampler for the mixed kernel (Supplemental C.4):
//   f(y) = sum_h w_h N(y_N | mu_h, Sigma_h)
//                prod_t Cat(y_Ct | psi_ht) prod_t' Pois(y_Pt' | rho_ht')
// Any of the three blocks may be empty (0 columns); the sampler then
// degenerates to the corresponding single-type or two-type model.
// [[Rcpp::export]]
List gibbs_mixed_cpp(
    const arma::mat& Y_N,  // n x qN, continuous responses (0 cols if absent)
    const arma::imat& Y_C, // n x qC, categorical responses in {1, ..., m_t}
    const arma::mat& Y_P,  // n x qP, count responses (non-negative integers)
    int k,
    int n_iter,
    int n_burnin,
    double alpha = -1,                                  // default: alpha = k
    double beta0 = 0.01,
    Nullable<NumericVector> nu0_ = R_NilValue,          // default: qN + 2
    Nullable<NumericVector> eps0_ = R_NilValue,         // default: qN + 2
    Nullable<NumericVector> alpha_c_ = R_NilValue,      // default: m_t
    Nullable<NumericVector> alpha_p_ = R_NilValue,      // default: rep(1, qP)
    Nullable<NumericVector> beta_p_ = R_NilValue        // default: rep(0.1, qP)
) {
  int qN = Y_N.n_cols;
  int qC = Y_C.n_cols;
  int qP = Y_P.n_cols;
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  int n_samples = n_iter - n_burnin;

  if (alpha < 0) alpha = (double)k;

  // --- Gaussian block priors ---
  double nu0 = nu0_.isNull() ? qN + 2 : as<NumericVector>(nu0_.get())[0];
  double eps0 = eps0_.isNull() ? qN + 2 : as<NumericVector>(eps0_.get())[0];

  // --- Categorical block prior (default alpha_c = m_t) ---
  ivec m(std::max(qC, 1), fill::zeros);
  for (int t = 0; t < qC; t++) {
    m(t) = Y_C.col(t).max();
  }
  vec alpha_c;
  if (alpha_c_.isNull()) {
    alpha_c = conv_to<vec>::from(m.head(std::max(qC, 1)));
  } else {
    alpha_c = as<vec>(alpha_c_.get());
  }

  // --- Poisson block priors ---
  vec alpha_p, beta_p;
  if (alpha_p_.isNull()) {
    alpha_p = vec(std::max(qP, 1), fill::ones);
  } else {
    alpha_p = as<vec>(alpha_p_.get());
  }
  if (beta_p_.isNull()) {
    beta_p = vec(std::max(qP, 1));
    beta_p.fill(0.1);
  } else {
    beta_p = as<vec>(beta_p_.get());
  }

  mixture::Model md(
    Y_N, Y_C, Y_P, k, alpha, beta0, nu0, eps0, alpha_c, alpha_p, beta_p
  );
  mixture::Store store(md, n_samples);
  mixture::Params par(md);

  // --- Initial values ---
  ivec z(n);
  for (int i = 0; i < n; i++) z(i) = i % k;

  // Gaussian block: means jittered around the sample mean
  if (qN > 0) {
    vec y_mean = arma::mean(Y_N, 0).t();
    mat S_y = cov(Y_N);
    mat L_init = chol(S_y / k, "lower");
    for (int h = 0; h < k; h++) {
      par.mu[h] = y_mean + L_init * randn(qN);
      par.Sigma[h] = S_y;
      par.L_chol[h] = chol(par.Sigma[h], "lower");
      par.log_det_Sigma(h) = 2.0 * accu(arma::log(par.L_chol[h].diag()));
      par.log_dens_N.col(h) = mixture::dmvnorm_log_from_chol(
        Y_N, par.mu[h], par.L_chol[h], par.log_det_Sigma(h)
      );
    }
  }

  // Categorical block: psi uniform (set by Params); Poisson block: prior draw
  for (int t = 0; t < qP; t++) {
    for (int h = 0; h < k; h++) {
      // randg uses scale parameterization, R's rgamma uses rate
      par.rho(h, t) = as_scalar(
        randg(1, distr_param(alpha_p(t), 1.0 / beta_p(t)))
      );
    }
  }

  // Kernel log-density matrix n x k (without weights), rebuilt each iteration
  mat log_dens(n, k);

  // --- Sampler ---
  for (int iter = 0; iter < n_iter; iter++) {
    // 0. Kernel log-densities at the current parameters
    mixture::kernel_log_dens(md, par, log_dens);

    // 1. Update cluster allocations z_i
    vec log_w = arma::log(par.w);
    mat log_p = log_dens;
    log_p.each_row() += log_w.t();
    vec row_max = max(log_p, 1);
    log_p.each_col() -= row_max;
    mat p_mat = exp(log_p);
    for (int i = 0; i < n; i++) {
      vec probs = p_mat.row(i).t();
      z(i) = mixture::sample_one(probs);
    }

    // 2-3. Update weights and kernel parameters given z
    ivec n_h = mixture::tabulate_vec(z + 1, k);
    mixture::draw_params(md, par, z, n_h);

    // 4. Store after burn-in
    if (iter >= n_burnin) {
      store.save(iter - n_burnin, z, par);
    }
  }

  return store.to_list();
}
