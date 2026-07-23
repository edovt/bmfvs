#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Sample one integer in {0, ..., k-1} with probabilities proportional to probs
static int sample_one(const vec& probs) {
  double u = randu<double>() * accu(probs);
  double cumsum = 0.0;
  for (uword i = 0; i < probs.n_elem; i++) {
    cumsum += probs(i);
    if (u <= cumsum) return i;
  }
  return probs.n_elem - 1;
}

// Draw from Dirichlet(alpha) using Armadillo's randg (Gamma distribution)
static vec rdirichlet_vec(const vec& alpha) {
  vec draw(alpha.n_elem);
  for (uword i = 0; i < alpha.n_elem; i++) {
    draw(i) = as_scalar(randg(1, distr_param(alpha(i), 1.0)));
  }
  return draw / accu(draw);
}

// Tabulate: count occurrences of values 1, ..., nbins in x
static ivec tabulate_vec(const ivec& x, int nbins) {
  ivec counts(nbins, fill::zeros);
  for (uword i = 0; i < x.n_elem; i++) {
    if (x(i) >= 1 && x(i) <= nbins) {
      counts(x(i) - 1)++;
    }
  }
  return counts;
}

// Compute log N(y_i | mu, Sigma) for each row of Y, given Cholesky
// L = chol(Sigma, "lower") and precomputed log_det = log|Sigma|
static vec dmvnorm_log_from_chol(
    const mat& Y, const vec& mu, const mat& L, double log_det
) {
  int q = Y.n_cols;
  double constant = -0.5 * q * std::log(2.0 * M_PI) - 0.5 * log_det;
  mat diff = Y.each_row() - mu.t();           // n x q
  mat V = solve(trimatl(L), diff.t());        // q x n via triangular solve
  return constant - 0.5 * sum(V % V, 0).t();  // n-vector
}

// Unified Gibbs sampler for the mixed kernel (Supplemental C.4):
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
    Nullable<IntegerVector> alpha_c_ = R_NilValue,      // default: m_t
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

  // --- Categorical block: number of levels per response and prior ---
  ivec m(std::max(qC, 1), fill::zeros);
  for (int t = 0; t < qC; t++) {
    m(t) = Y_C.col(t).max();
  }
  ivec alpha_c;
  if (alpha_c_.isNull()) {
    alpha_c = m.head(std::max(qC, 1));
  } else {
    alpha_c = as<ivec>(alpha_c_.get());
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

  // --- Storage ---
  imat z_store(n_samples, n);
  mat w_store(n_samples, k);
  cube mu_store(n_samples, k, std::max(qN, 1), fill::zeros);
  NumericVector Sigma_store(n_samples * k * std::max(qN, 1) * std::max(qN, 1));
  Sigma_store.attr("dim") = IntegerVector::create(
    n_samples, k, std::max(qN, 1), std::max(qN, 1)
  );
  std::vector<cube> psi_store(qC);
  for (int t = 0; t < qC; t++) {
    psi_store[t] = cube(n_samples, k, m(t), fill::zeros);
  }
  cube rho_store(n_samples, k, std::max(qP, 1), fill::zeros);
  mat log_lik_store(n_samples, n, fill::zeros);

  // --- Initial values ---
  ivec z(n);
  for (int i = 0; i < n; i++) z(i) = i % k;
  vec w(k, fill::ones);
  w /= k;

  // Gaussian block state (cached Cholesky and log-densities)
  std::vector<vec> mu(k);
  std::vector<mat> Sigma(k);
  std::vector<mat> L_chol(k);
  vec log_det_Sigma(k, fill::zeros);
  mat log_dens_N(n, k, fill::zeros);
  if (qN > 0) {
    vec y_mean = arma::mean(Y_N, 0).t();
    mat S_y = cov(Y_N);
    mat L_init = chol(S_y / k, "lower");
    for (int h = 0; h < k; h++) {
      mu[h] = y_mean + L_init * randn(qN);
      Sigma[h] = S_y;
      L_chol[h] = chol(Sigma[h], "lower");
      log_det_Sigma(h) = 2.0 * accu(arma::log(L_chol[h].diag()));
      log_dens_N.col(h) = dmvnorm_log_from_chol(
        Y_N, mu[h], L_chol[h], log_det_Sigma(h)
      );
    }
  }

  // Categorical block state: psi[t] is a k x m(t) matrix
  std::vector<mat> psi(qC);
  for (int t = 0; t < qC; t++) {
    psi[t] = mat(k, m(t));
    psi[t].fill(1.0 / m(t));
  }

  // Poisson block state: rho is k x qP
  mat rho(k, std::max(qP, 1));
  for (int t = 0; t < qP; t++) {
    for (int h = 0; h < k; h++) {
      // randg uses scale parameterization, R's rgamma uses rate
      rho(h, t) = as_scalar(
        randg(1, distr_param(alpha_p(t), 1.0 / beta_p(t)))
      );
    }
  }

  // Precompute -sum_t lgamma(y_it + 1) for the Poisson log-likelihood
  vec log_fact_P(n, fill::zeros);
  if (qP > 0) {
    for (int i = 0; i < n; i++) {
      for (int t = 0; t < qP; t++) {
        log_fact_P(i) -= std::lgamma(Y_P(i, t) + 1.0);
      }
    }
  }

  // Kernel log-density matrix n x k (without weights), rebuilt each iteration
  mat log_dens(n, k);

  // --- Sampler ---
  for (int iter = 0; iter < n_iter; iter++) {
    // 0. Assemble kernel log-densities: sum over the three blocks
    log_dens.zeros();
    if (qN > 0) log_dens += log_dens_N;
    if (qC > 0) {
      for (int h = 0; h < k; h++) {
        for (int t = 0; t < qC; t++) {
          vec log_psi_ht = arma::log(psi[t].row(h).t());
          for (int i = 0; i < n; i++) {
            log_dens(i, h) += log_psi_ht(Y_C(i, t) - 1);
          }
        }
      }
    }
    if (qP > 0) {
      mat log_rho = arma::log(rho.head_cols(qP)); // k x qP
      for (int h = 0; h < k; h++) {
        double rho_sum_h = accu(rho.row(h).head(qP));
        log_dens.col(h) += Y_P * log_rho.row(h).t() - rho_sum_h;
      }
      // (lgamma(y+1) terms are constant across h; added only to log_lik)
    }

    // 1. Update cluster allocations z_i
    vec log_w = arma::log(w);
    mat log_p = log_dens;
    log_p.each_row() += log_w.t();
    vec row_max = max(log_p, 1);
    log_p.each_col() -= row_max;
    mat p_mat = exp(log_p);
    for (int i = 0; i < n; i++) {
      vec probs = p_mat.row(i).t();
      z(i) = sample_one(probs);
    }

    // 2. Update weights
    ivec n_h = tabulate_vec(z + 1, k);
    vec dir_alpha(k);
    for (int h = 0; h < k; h++) dir_alpha(h) = alpha / k + n_h(h);
    w = rdirichlet_vec(dir_alpha);

    // 3a. Update Gaussian block (mu_h, Sigma_h) and refresh log_dens_N
    if (qN > 0) {
      for (int h = 0; h < k; h++) {
        int nh = n_h(h);
        if (nh == 0) {
          Sigma[h] = iwishrnd(eps0 * eye(qN, qN), nu0);
          L_chol[h] = chol(Sigma[h], "lower");
          // mu ~ N(0, Sigma / beta0) so chol(Sigma/beta0) = L / sqrt(beta0)
          mu[h] = (L_chol[h] / std::sqrt(beta0)) * randn(qN);
        } else {
          uvec idx = find(z == h);
          mat Y_h = Y_N.rows(idx);
          vec y_bar = arma::mean(Y_h, 0).t();
          mat centered = Y_h.each_row() - y_bar.t();
          mat S_h = centered.t() * centered;
          mat Psi_post = eps0 * eye(qN, qN) + S_h +
                         (beta0 * nh / (beta0 + nh)) * (y_bar * y_bar.t());
          Sigma[h] = iwishrnd(Psi_post, nu0 + nh);
          L_chol[h] = chol(Sigma[h], "lower");
          vec mu_post = (nh / (beta0 + nh)) * y_bar;
          mu[h] = mu_post + (L_chol[h] / std::sqrt(beta0 + nh)) * randn(qN);
        }
        log_det_Sigma(h) = 2.0 * accu(arma::log(L_chol[h].diag()));
        log_dens_N.col(h) = dmvnorm_log_from_chol(
          Y_N, mu[h], L_chol[h], log_det_Sigma(h)
        );
      }
    }

    // 3b. Update categorical block psi_ht
    for (int t = 0; t < qC; t++) {
      for (int h = 0; h < k; h++) {
        ivec counts(m(t), fill::zeros);
        for (int i = 0; i < n; i++) {
          if (z(i) == h) {
            counts(Y_C(i, t) - 1)++;
          }
        }
        vec dir_alpha_t(m(t));
        for (int s = 0; s < m(t); s++) {
          dir_alpha_t(s) = (double)alpha_c(t) / m(t) + counts(s);
        }
        psi[t].row(h) = rdirichlet_vec(dir_alpha_t).t();
      }
    }

    // 3c. Update Poisson block rho_ht ~ Ga(alpha_p[t] + S_ht, beta_p[t] + n_h)
    if (qP > 0) {
      for (int h = 0; h < k; h++) {
        for (int t = 0; t < qP; t++) {
          double S_ht = 0.0;
          for (int i = 0; i < n; i++) {
            if (z(i) == h) {
              S_ht += Y_P(i, t);
            }
          }
          double shape = alpha_p(t) + S_ht;
          double scale = 1.0 / (beta_p(t) + n_h(h));
          rho(h, t) = as_scalar(randg(1, distr_param(shape, scale)));
        }
      }
    }

    // 4. Store after burn-in
    if (iter >= n_burnin) {
      int s = iter - n_burnin;
      z_store.row(s) = z.t() + 1; // back to 1-indexed for R
      w_store.row(s) = w.t();

      if (qN > 0) {
        for (int h = 0; h < k; h++) {
          mu_store.tube(s, h) = mu[h];
          const mat& Sh = Sigma[h];
          for (int b = 0; b < qN; b++) {
            for (int a = 0; a < qN; a++) {
              int flat = s + n_samples * (h + k * (a + qN * b));
              Sigma_store[flat] = Sh(a, b);
            }
          }
        }
      }
      for (int t = 0; t < qC; t++) {
        for (int h = 0; h < k; h++) {
          psi_store[t].tube(s, h) = psi[t].row(h).t();
        }
      }
      if (qP > 0) {
        for (int h = 0; h < k; h++) {
          rho_store.tube(s, h) = rho.row(h).head(qP).t();
        }
      }

      // Log-likelihood: recompute kernel log-densities with the new
      // parameters (blocks were updated after z), then mix over h
      mat log_ll = mat(n, k, fill::zeros);
      if (qN > 0) log_ll += log_dens_N; // already refreshed in step 3a
      if (qC > 0) {
        for (int h = 0; h < k; h++) {
          for (int t = 0; t < qC; t++) {
            vec log_psi_ht = arma::log(psi[t].row(h).t());
            for (int i = 0; i < n; i++) {
              log_ll(i, h) += log_psi_ht(Y_C(i, t) - 1);
            }
          }
        }
      }
      if (qP > 0) {
        mat log_rho = arma::log(rho.head_cols(qP));
        for (int h = 0; h < k; h++) {
          double rho_sum_h = accu(rho.row(h).head(qP));
          log_ll.col(h) += Y_P * log_rho.row(h).t() - rho_sum_h;
        }
        log_ll.each_col() += log_fact_P;
      }

      vec log_w_new = arma::log(w);
      log_ll.each_row() += log_w_new.t();
      vec row_max_ll = max(log_ll, 1);
      log_ll.each_col() -= row_max_ll;
      vec ll = row_max_ll + arma::log(sum(exp(log_ll), 1));
      log_lik_store.row(s) = ll.t();
    }
  }

  // --- Return only the blocks that are present ---
  List out = List::create(
    Named("z") = z_store,
    Named("w") = w_store,
    Named("log_lik") = log_lik_store
  );
  if (qN > 0) {
    out["mu"] = mu_store;
    out["Sigma"] = Sigma_store;
  }
  if (qC > 0) {
    List psi_list(qC);
    for (int t = 0; t < qC; t++) {
      psi_list[t] = psi_store[t];
    }
    out["psi"] = psi_list;
  }
  if (qP > 0) {
    out["rho"] = rho_store;
  }
  return out;
}
