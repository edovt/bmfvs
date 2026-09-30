#ifndef BMFVS_MIXTURE_H
#define BMFVS_MIXTURE_H

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <vector>

// Pieces of the mixed-kernel Gibbs sampler (Supplemental C.4) shared by all
// Stage-1 samplers: the conjugate draws of (w, theta) | z, the kernel
// log-densities and the storage of the posterior samples. The conditional
// sampler alternates z | (w, theta) with these draws; the collapsed samplers
// update z with theta integrated out and call them only at saved sweeps.
namespace mixture {

// Sample one integer in {0, ..., k-1} with probabilities proportional to probs
inline int sample_one(const arma::vec& probs) {
  double u = arma::randu<double>() * arma::accu(probs);
  double cumsum = 0.0;
  for (arma::uword i = 0; i < probs.n_elem; i++) {
    cumsum += probs(i);
    if (u <= cumsum) return i;
  }
  return probs.n_elem - 1;
}

// Draw from Dirichlet(alpha) using Armadillo's randg (Gamma distribution)
inline arma::vec rdirichlet_vec(const arma::vec& alpha) {
  arma::vec draw(alpha.n_elem);
  for (arma::uword i = 0; i < alpha.n_elem; i++) {
    draw(i) = arma::as_scalar(arma::randg(1, arma::distr_param(alpha(i), 1.0)));
  }
  return draw / arma::accu(draw);
}

// Tabulate: count occurrences of values 1, ..., nbins in x
inline arma::ivec tabulate_vec(const arma::ivec& x, int nbins) {
  arma::ivec counts(nbins, arma::fill::zeros);
  for (arma::uword i = 0; i < x.n_elem; i++) {
    if (x(i) >= 1 && x(i) <= nbins) {
      counts(x(i) - 1)++;
    }
  }
  return counts;
}

// Compute log N(y_i | mu, Sigma) for each row of Y, given Cholesky
// L = chol(Sigma, "lower") and precomputed log_det = log|Sigma|
inline arma::vec dmvnorm_log_from_chol(
    const arma::mat& Y, const arma::vec& mu, const arma::mat& L, double log_det
) {
  int q = Y.n_cols;
  double constant = -0.5 * q * std::log(2.0 * M_PI) - 0.5 * log_det;
  arma::mat diff = Y.each_row() - mu.t();                    // n x q
  arma::mat V = arma::solve(arma::trimatl(L), diff.t());     // q x n
  return constant - 0.5 * arma::sum(V % V, 0).t();           // n-vector
}

// Data, prior hyperparameters and dimensions (defaults already resolved)
struct Model {
  const arma::mat& Y_N;  // n x qN, continuous responses (0 cols if absent)
  const arma::imat& Y_C; // n x qC, categorical responses in {1, ..., m_t}
  const arma::mat& Y_P;  // n x qP, count responses
  int n, k, qN, qC, qP;
  double alpha, beta0, nu0, eps0;
  arma::ivec m;          // levels per categorical response
  arma::vec alpha_c, alpha_p, beta_p;
  arma::vec log_fact_P;  // -sum_t lgamma(y_it + 1), for the log-likelihood

  Model(const arma::mat& Y_N_, const arma::imat& Y_C_, const arma::mat& Y_P_,
        int k_, double alpha_, double beta0_, double nu0_, double eps0_,
        const arma::vec& alpha_c_, const arma::vec& alpha_p_,
        const arma::vec& beta_p_)
      : Y_N(Y_N_), Y_C(Y_C_), Y_P(Y_P_), k(k_), alpha(alpha_), beta0(beta0_),
        nu0(nu0_), eps0(eps0_), alpha_c(alpha_c_), alpha_p(alpha_p_),
        beta_p(beta_p_) {
    qN = Y_N.n_cols;
    qC = Y_C.n_cols;
    qP = Y_P.n_cols;
    n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
    m = arma::ivec(std::max(qC, 1), arma::fill::zeros);
    for (int t = 0; t < qC; t++) {
      m(t) = Y_C.col(t).max();
    }
    log_fact_P = arma::vec(n, arma::fill::zeros);
    if (qP > 0) {
      for (int i = 0; i < n; i++) {
        for (int t = 0; t < qP; t++) {
          log_fact_P(i) -= std::lgamma(Y_P(i, t) + 1.0);
        }
      }
    }
  }
};

// Component weights and kernel parameters, plus the cached Gaussian
// Cholesky factors and log-densities
struct Params {
  arma::vec w;
  std::vector<arma::vec> mu;
  std::vector<arma::mat> Sigma;
  std::vector<arma::mat> L_chol;
  arma::vec log_det_Sigma;
  arma::mat log_dens_N;          // n x k
  std::vector<arma::mat> psi;    // psi[t] is k x m(t)
  arma::mat rho;                 // k x qP

  explicit Params(const Model& md)
      : mu(md.k), Sigma(md.k), L_chol(md.k), psi(md.qC) {
    w = arma::vec(md.k, arma::fill::ones);
    w /= md.k;
    log_det_Sigma = arma::vec(md.k, arma::fill::zeros);
    log_dens_N = arma::mat(md.n, md.k, arma::fill::zeros);
    for (int t = 0; t < md.qC; t++) {
      psi[t] = arma::mat(md.k, md.m(t));
      psi[t].fill(1.0 / md.m(t));
    }
    rho = arma::mat(md.k, std::max(md.qP, 1));
  }
};

// Kernel log-densities n x k (without weights and without the log(y!) terms)
inline void kernel_log_dens(const Model& md, const Params& par, arma::mat& out) {
  out.zeros();
  if (md.qN > 0) out += par.log_dens_N;
  if (md.qC > 0) {
    for (int h = 0; h < md.k; h++) {
      for (int t = 0; t < md.qC; t++) {
        arma::vec log_psi_ht = arma::log(par.psi[t].row(h).t());
        for (int i = 0; i < md.n; i++) {
          out(i, h) += log_psi_ht(md.Y_C(i, t) - 1);
        }
      }
    }
  }
  if (md.qP > 0) {
    arma::mat log_rho = arma::log(par.rho.head_cols(md.qP)); // k x qP
    for (int h = 0; h < md.k; h++) {
      double rho_sum_h = arma::accu(par.rho.row(h).head(md.qP));
      out.col(h) += md.Y_P * log_rho.row(h).t() - rho_sum_h;
    }
  }
}

// Draw (w, theta) | z from the conjugate full conditionals; z is 0-based and
// n_h its cluster sizes
inline void draw_params(
    const Model& md, Params& par, const arma::ivec& z, const arma::ivec& n_h
) {
  const int k = md.k, qN = md.qN, n = md.n;

  // Weights
  arma::vec dir_alpha(k);
  for (int h = 0; h < k; h++) dir_alpha(h) = md.alpha / k + n_h(h);
  par.w = rdirichlet_vec(dir_alpha);

  // Gaussian block (mu_h, Sigma_h), refreshing the cached log-densities
  if (qN > 0) {
    for (int h = 0; h < k; h++) {
      int nh = n_h(h);
      if (nh == 0) {
        par.Sigma[h] = arma::iwishrnd(md.eps0 * arma::eye(qN, qN), md.nu0);
        par.L_chol[h] = arma::chol(par.Sigma[h], "lower");
        // mu ~ N(0, Sigma / beta0) so chol(Sigma/beta0) = L / sqrt(beta0)
        par.mu[h] = (par.L_chol[h] / std::sqrt(md.beta0)) * arma::randn(qN);
      } else {
        arma::uvec idx = arma::find(z == h);
        arma::mat Y_h = md.Y_N.rows(idx);
        arma::vec y_bar = arma::mean(Y_h, 0).t();
        arma::mat centered = Y_h.each_row() - y_bar.t();
        arma::mat S_h = centered.t() * centered;
        arma::mat Psi_post = md.eps0 * arma::eye(qN, qN) + S_h +
                             (md.beta0 * nh / (md.beta0 + nh)) * (y_bar * y_bar.t());
        par.Sigma[h] = arma::iwishrnd(Psi_post, md.nu0 + nh);
        par.L_chol[h] = arma::chol(par.Sigma[h], "lower");
        arma::vec mu_post = (nh / (md.beta0 + nh)) * y_bar;
        par.mu[h] = mu_post +
                    (par.L_chol[h] / std::sqrt(md.beta0 + nh)) * arma::randn(qN);
      }
      par.log_det_Sigma(h) = 2.0 * arma::accu(arma::log(par.L_chol[h].diag()));
      par.log_dens_N.col(h) = dmvnorm_log_from_chol(
        md.Y_N, par.mu[h], par.L_chol[h], par.log_det_Sigma(h)
      );
    }
  }

  // Categorical block psi_ht
  for (int t = 0; t < md.qC; t++) {
    for (int h = 0; h < k; h++) {
      arma::ivec counts(md.m(t), arma::fill::zeros);
      for (int i = 0; i < n; i++) {
        if (z(i) == h) {
          counts(md.Y_C(i, t) - 1)++;
        }
      }
      arma::vec dir_alpha_t(md.m(t));
      for (int s = 0; s < md.m(t); s++) {
        dir_alpha_t(s) = md.alpha_c(t) / md.m(t) + counts(s);
      }
      par.psi[t].row(h) = rdirichlet_vec(dir_alpha_t).t();
    }
  }

  // Poisson block rho_ht ~ Ga(alpha_p[t] + S_ht, beta_p[t] + n_h)
  if (md.qP > 0) {
    for (int h = 0; h < k; h++) {
      for (int t = 0; t < md.qP; t++) {
        double S_ht = 0.0;
        for (int i = 0; i < n; i++) {
          if (z(i) == h) {
            S_ht += md.Y_P(i, t);
          }
        }
        double shape = md.alpha_p(t) + S_ht;
        double scale = 1.0 / (md.beta_p(t) + n_h(h));
        // randg uses scale parameterization, R's rgamma uses rate
        par.rho(h, t) = arma::as_scalar(
          arma::randg(1, arma::distr_param(shape, scale))
        );
      }
    }
  }
}

// Storage of the saved posterior samples, returned to R in the format
// consumed by Stage 2, the hyperparameter selection and the WAIC step
class Store {
 public:
  Store(const Model& md, int n_samples)
      : md_(md), n_samples_(n_samples), psi_(md.qC), log_ll_(md.n, md.k) {
    const int k = md.k, qN1 = std::max(md.qN, 1);
    z_ = arma::imat(n_samples, md.n);
    w_ = arma::mat(n_samples, k);
    mu_ = arma::cube(n_samples, k, qN1, arma::fill::zeros);
    Sigma_ = Rcpp::NumericVector(n_samples * k * qN1 * qN1);
    Sigma_.attr("dim") = Rcpp::IntegerVector::create(n_samples, k, qN1, qN1);
    for (int t = 0; t < md.qC; t++) {
      psi_[t] = arma::cube(n_samples, k, md.m(t), arma::fill::zeros);
    }
    rho_ = arma::cube(n_samples, k, std::max(md.qP, 1), arma::fill::zeros);
    log_lik_ = arma::mat(n_samples, md.n, arma::fill::zeros);
  }

  // Save sample s; z is 0-based
  void save(int s, const arma::ivec& z, const Params& par) {
    const int k = md_.k, qN = md_.qN;
    z_.row(s) = z.t() + 1; // back to 1-indexed for R
    w_.row(s) = par.w.t();

    if (qN > 0) {
      for (int h = 0; h < k; h++) {
        mu_.tube(s, h) = par.mu[h];
        const arma::mat& Sh = par.Sigma[h];
        for (int b = 0; b < qN; b++) {
          for (int a = 0; a < qN; a++) {
            int flat = s + n_samples_ * (h + k * (a + qN * b));
            Sigma_[flat] = Sh(a, b);
          }
        }
      }
    }
    for (int t = 0; t < md_.qC; t++) {
      for (int h = 0; h < k; h++) {
        psi_[t].tube(s, h) = par.psi[t].row(h).t();
      }
    }
    if (md_.qP > 0) {
      for (int h = 0; h < k; h++) {
        rho_.tube(s, h) = par.rho.row(h).head(md_.qP).t();
      }
    }

    // Pointwise log-likelihood log sum_h w_h K(y_i | theta_h) at the saved
    // parameters (used by WAIC)
    kernel_log_dens(md_, par, log_ll_);
    if (md_.qP > 0) log_ll_.each_col() += md_.log_fact_P;
    arma::vec log_w = arma::log(par.w);
    log_ll_.each_row() += log_w.t();
    arma::vec row_max = arma::max(log_ll_, 1);
    log_ll_.each_col() -= row_max;
    arma::vec ll = row_max + arma::log(arma::sum(arma::exp(log_ll_), 1));
    log_lik_.row(s) = ll.t();
  }

  // Return only the blocks that are present
  Rcpp::List to_list() const {
    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("z") = z_,
      Rcpp::Named("w") = w_,
      Rcpp::Named("log_lik") = log_lik_
    );
    if (md_.qN > 0) {
      out["mu"] = mu_;
      out["Sigma"] = Sigma_;
    }
    if (md_.qC > 0) {
      Rcpp::List psi_list(md_.qC);
      for (int t = 0; t < md_.qC; t++) {
        psi_list[t] = psi_[t];
      }
      out["psi"] = psi_list;
    }
    if (md_.qP > 0) {
      out["rho"] = rho_;
    }
    return out;
  }

 private:
  const Model& md_;
  int n_samples_;
  arma::imat z_;
  arma::mat w_;
  arma::cube mu_;
  Rcpp::NumericVector Sigma_;
  std::vector<arma::cube> psi_;
  arma::cube rho_;
  arma::mat log_lik_;
  arma::mat log_ll_; // n x k workspace
};

} // namespace mixture

#endif
