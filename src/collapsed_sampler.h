#ifndef BMFVS_COLLAPSED_SAMPLER_H
#define BMFVS_COLLAPSED_SAMPLER_H

#include "collapsed.h"

// Samplers on the collapsed posterior of the allocations,
//   pi(z | Y) ∝ prod_h Gamma(alpha/k + n_h) m(Y_h),
// (Ascolani & Zanella, 2025, arXiv:2510.03226):
//   - marginal Gibbs (P_MG): deterministic scan of the full conditionals;
//   - non-reversible lifted sampler (P_NR, their Algorithms 4-6): each step
//     picks a pair of clusters and proposes moving one point from one to the
//     other in the direction of the pair's velocity, which is reversed only
//     on rejection (or at rate xi / n).
// All random numbers come from R's generator, so set.seed() reproduces runs.
namespace collapsed {

class AllocSampler {
 public:
  AllocSampler(const Prior& pr, const Data& d, int k, double a_k, double xi)
      : pr_(pr), d_(d), n_(d.n), k_(k), a_k_(a_k), xi_(xi),
        work_(std::max(pr.qN * pr.qN, 1)), lp_(k), LN_new_(k) {}

  // z_i ~ Unif{1, ..., k}; for the lifted sampler also one velocity per pair
  // of clusters, uniform on {-1, +1}
  void init(bool lifted) {
    z_.set_size(n_);
    for (int i = 0; i < n_; i++) z_(i) = unif_index(k_);
    vel_.zeros(k_, k_);
    if (lifted) {
      for (int a = 0; a < k_; a++) {
        for (int b = a + 1; b < k_; b++) vel_(a, b) = R::unif_rand() < 0.5 ? 1 : -1;
      }
    }
    members_.assign(k_, std::vector<int>());
    pos_.assign(n_, 0);
    for (int i = 0; i < n_; i++) {
      pos_[i] = members_[z_(i)].size();
      members_[z_(i)].push_back(i);
    }
    resync();
    reset_counters();
  }

  // Recompute the statistics and the cached Gaussian log m from z, clearing
  // the floating-point drift of repeated additions and removals
  void resync() {
    st_ = stats_from_z(pr_, d_, z_, k_);
    LN_.assign(k_, 0.0);
    for (int h = 0; h < k_; h++) {
      LN_[h] = log_m_gauss(pr_, st_[h], nullptr, 0, work_.data());
    }
  }

  // P_MG: resample z_i from pi(z_i | z_-i, Y)
  void update_site(int i) {
    const int h0 = z_(i);
    double mx = -INFINITY;
    for (int h = 0; h < k_; h++) {
      const bool in = (h == h0);
      lp_[h] = std::log(a_k_ + st_[h].n - in) +
               log_pred(pr_, d_, st_[h], LN_[h], i, in, work_.data(), &LN_new_[h]);
      mx = std::max(mx, lp_[h]);
    }
    double total = 0.0;
    for (int h = 0; h < k_; h++) {
      lp_[h] = std::exp(lp_[h] - mx);
      total += lp_[h];
    }
    double u = R::unif_rand() * total, cum = 0.0;
    int h_new = k_ - 1;
    for (int h = 0; h < k_; h++) {
      cum += lp_[h];
      if (u <= cum) {
        h_new = h;
        break;
      }
    }
    n_steps_++;
    if (h_new != h0) {
      move(i, h0, h_new, LN_new_[h0], LN_new_[h_new]);
      n_accepted_++;
    }
  }

  void sweep_marginal() {
    if (k_ < 2) return;
    for (int i = 0; i < n_; i++) update_site(i);
  }

  // P_NR: one lifted Metropolis-Hastings step (Algorithms 4-5)
  void step_nonreversible() {
    // Pair (k1, k2) with probability (n_k1 + n_k2) / ((k - 1) n): k1 is the
    // cluster of a uniformly drawn point, k2 uniform among the others
    const int k1 = z_(unif_index(n_));
    int k2 = unif_index(k_ - 1);
    if (k2 >= k1) k2++;
    const int a = std::min(k1, k2), b = std::max(k1, k2);
    int& v = vel_(a, b);
    const double refresh = xi_ / n_;

    n_steps_++;
    if (R::unif_rand() < refresh) flip(v);
    const int km = v == 1 ? a : b;  // move a point from km ...
    const int kp = v == 1 ? b : a;  // ... to kp
    const int n_m = st_[km].n, n_p = st_[kp].n;
    if (n_m == 0) {
      flip(v);
    } else {
      n_proposed_++;
      const int i = members_[km][unif_index(n_m)];
      double LN_m, LN_p;
      double lp_plus = log_pred(pr_, d_, st_[kp], LN_[kp], i, false, work_.data(), &LN_p);
      double lp_minus = log_pred(pr_, d_, st_[km], LN_[km], i, true, work_.data(), &LN_m);
      // Proposal correction, Dirichlet prior on the sizes, predictive ratio
      double log_r = std::log((double)n_m) - std::log(n_p + 1.0) +
                     std::log(a_k_ + n_p) - std::log(a_k_ + n_m - 1.0) +
                     lp_plus - lp_minus;
      if (std::log(R::unif_rand()) < log_r) {
        move(i, km, kp, LN_m, LN_p);
        n_accepted_++;
      } else {
        flip(v);
      }
    }
    if (R::unif_rand() < refresh) flip(v);
  }

  // One sweep = n steps (n single-point proposals, not a scan)
  void sweep_nonreversible() {
    if (k_ < 2) return;
    for (int s = 0; s < n_; s++) step_nonreversible();
  }

  // Largest discrepancy between the cached state and a recomputation from z
  // (integer quantities, floating-point statistics, cached Gaussian log m,
  // and whether the member lists agree with z)
  Rcpp::List cache_error() const {
    std::vector<Stats> ref = stats_from_z(pr_, d_, z_, k_);
    int int_err = 0;
    double stat_err = 0.0, LN_err = 0.0;
    bool members_ok = true;
    std::vector<double> work(work_.size());
    for (int h = 0; h < k_; h++) {
      int_err = std::max(int_err, std::abs(ref[h].n - st_[h].n));
      if (pr_.n_levels > 0) {
        int_err = std::max(int_err, (int)arma::max(arma::abs(ref[h].cnt - st_[h].cnt)));
      }
      if (pr_.qN > 0) {
        stat_err = std::max(stat_err, arma::abs(ref[h].s - st_[h].s).max());
        stat_err = std::max(stat_err, arma::abs(arma::trimatl(ref[h].Q - st_[h].Q)).max());
      }
      if (pr_.qP > 0) {
        stat_err = std::max(stat_err, arma::abs(ref[h].S - st_[h].S).max());
      }
      stat_err = std::max(stat_err, std::abs(ref[h].lfact - st_[h].lfact));
      double LN_ref = log_m_gauss(pr_, ref[h], nullptr, 0, work.data());
      LN_err = std::max(LN_err, std::abs(LN_ref - LN_[h]));
      if ((int)members_[h].size() != ref[h].n) members_ok = false;
      for (size_t p = 0; p < members_[h].size(); p++) {
        int i = members_[h][p];
        if (z_(i) != h || pos_[i] != (int)p) members_ok = false;
      }
    }
    return Rcpp::List::create(
      Rcpp::Named("int_err") = int_err,
      Rcpp::Named("stat_err") = stat_err,
      Rcpp::Named("LN_err") = LN_err,
      Rcpp::Named("members_ok") = members_ok
    );
  }

  const arma::ivec& z() const { return z_; }

  arma::ivec counts() const {
    arma::ivec out(k_);
    for (int h = 0; h < k_; h++) out(h) = st_[h].n;
    return out;
  }

  void reset_counters() {
    n_steps_ = n_proposed_ = n_accepted_ = n_flips_ = 0.0;
  }

  // Lifted sampler: acceptance rate among proposals from non-empty clusters,
  // and mean excursion length (steps on a pair between velocity reversals).
  // Marginal sampler: fraction of site updates that changed z_i.
  Rcpp::List diagnostics(bool lifted) const {
    if (lifted) {
      return Rcpp::List::create(
        Rcpp::Named("acceptance_rate") = n_accepted_ / n_proposed_,
        Rcpp::Named("mean_excursion") = n_steps_ / n_flips_
      );
    }
    return Rcpp::List::create(Rcpp::Named("move_rate") = n_accepted_ / n_steps_);
  }

 private:
  const Prior& pr_;
  const Data& d_;
  const int n_, k_;
  const double a_k_, xi_;
  arma::ivec z_;
  std::vector<Stats> st_;
  std::vector<double> LN_;                  // Gaussian log m per cluster
  std::vector<std::vector<int>> members_;   // members of each cluster
  std::vector<int> pos_;                    // position of i in members_[z_i]
  arma::imat vel_;                          // velocity of pair (a, b), a < b
  std::vector<double> work_, lp_, LN_new_;
  double n_steps_, n_proposed_, n_accepted_, n_flips_;

  // Uniform integer in {0, ..., m - 1} (unif_rand() lies in (0, 1))
  static int unif_index(int m) {
    return std::min((int)(m * R::unif_rand()), m - 1);
  }

  void flip(int& v) {
    v = -v;
    n_flips_++;
  }

  // Move subject i between clusters, with the new cached Gaussian log m's
  void move(int i, int from, int to, double LN_from, double LN_to) {
    update_stats(st_[from], d_, i, -1);
    update_stats(st_[to], d_, i, +1);
    std::vector<int>& mf = members_[from];
    int last = mf.back();
    mf[pos_[i]] = last;
    pos_[last] = pos_[i];
    mf.pop_back();
    pos_[i] = members_[to].size();
    members_[to].push_back(i);
    z_(i) = to;
    LN_[from] = LN_from;
    LN_[to] = LN_to;
  }
};

} // namespace collapsed

#endif
