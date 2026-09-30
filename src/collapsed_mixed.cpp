#include "collapsed_sampler.h"
#include "mixture.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Collapsed samplers for the mixed kernel: the allocations z are updated with
// (w, theta) integrated out ("marginal" = P_MG, "nonreversible" = P_NR), and
// (w, theta) | z is drawn from its conjugate full conditional at every saved
// sweep, so the saved (z, w, theta) are draws from the joint posterior.
// One iteration is one sweep of n single-point updates. The output has the
// same format as gibbs_mixed_cpp(), with sampler diagnostics (over the saved
// sweeps) in the "diagnostics" attribute.
// [[Rcpp::export]]
List collapsed_mixed_cpp(
    const arma::mat& Y_N,  // n x qN, continuous responses (0 cols if absent)
    const arma::imat& Y_C, // n x qC, categorical responses in {1, ..., m_t}
    const arma::mat& Y_P,  // n x qP, count responses (non-negative integers)
    int k,
    int n_iter,
    int n_burnin,
    double alpha,          // < 0: alpha = k
    const List& pp,        // priors from expand_prior_params()
    std::string sampler,   // "nonreversible" or "marginal"
    double xi = 0.5        // P_NR refresh rate (Ascolani & Zanella's choice)
) {
  bool lifted;
  if (sampler == "nonreversible") {
    lifted = true;
  } else if (sampler == "marginal") {
    lifted = false;
  } else {
    stop("Unknown collapsed sampler: " + sampler);
  }
  if (alpha < 0) alpha = (double)k;
  int n = std::max(Y_N.n_rows, std::max(Y_C.n_rows, Y_P.n_rows));
  int n_samples = n_iter - n_burnin;

  collapsed::Prior pr = collapsed::make_prior(Y_C, Y_N.n_cols, Y_P.n_cols, n, pp);
  collapsed::Data d = collapsed::make_data(Y_N, Y_C, Y_P, pr);
  mixture::Model md(
    Y_N, Y_C, Y_P, k, alpha, pr.beta0, pr.nu0, pr.eps0, pr.alpha_c,
    pr.alpha_p, pr.beta_p
  );
  mixture::Store store(md, n_samples);
  mixture::Params par(md);

  collapsed::AllocSampler smp(pr, d, k, alpha / k, xi);
  smp.init(lifted);

  for (int iter = 0; iter < n_iter; iter++) {
    if (iter == n_burnin) smp.reset_counters();
    if (lifted) {
      smp.sweep_nonreversible();
    } else {
      smp.sweep_marginal();
    }
    smp.resync();

    if (iter >= n_burnin) {
      mixture::draw_params(md, par, smp.z(), smp.counts());
      store.save(iter - n_burnin, smp.z(), par);
    }
  }

  List out = store.to_list();
  out.attr("diagnostics") = smp.diagnostics(lifted);
  return out;
}
