# Posterior weights of (H_0j, H_1j1, H_1j2, H_1j3) for every (sample,
# predictor) pair, i.e. the summands of Phi in (2.10). b1, b2, b12 are the
# flattened log Bayes factors; the four returned vectors are the normalized
# kappa_t * BF_jt terms.
#
# pmax.int() rather than apply(x, 2, max): the columns number S * p, which is
# tens of millions in a screening problem, and this runs inside the fixed
# point iteration.
bf_weights <- function(log_kappa, b1, b2, b12) {
  t1 <- log_kappa[1]
  t2 <- log_kappa[2] + b1
  t3 <- log_kappa[3] + b2
  t4 <- log_kappa[4] + b12
  m <- pmax.int(t1, t2, t3, t4)
  e1 <- exp(t1 - m)
  e2 <- exp(t2 - m)
  e3 <- exp(t3 - m)
  e4 <- exp(t4 - m)
  denom <- e1 + e2 + e3 + e4
  list(w0 = e1 / denom, w1 = e2 / denom, w2 = e3 / denom, w3 = e4 / denom)
}

# Empirical Bayes estimation of the prior probabilities kappa (2.9), as the
# fixed point of Phi in (2.10).
estimate_kappa <- function(
  log_bf1,
  log_bf2,
  kappa_init = c(1 / 2, 1 / 6, 1 / 6, 1 / 6),
  max_iter = 100,
  tol = 1e-4,
  floor_kappa = 1e-12
) {
  N <- length(log_bf1)

  # Flattened once: the fixed point iteration only changes kappa
  b1 <- as.vector(log_bf1)
  b2 <- as.vector(log_bf2)
  b12 <- b1 + b2

  kappa <- kappa_init

  for (iter in seq_len(max_iter)) {
    wts <- bf_weights(log(kappa), b1, b2, b12)
    kappa_new <- c(sum(wts$w0), sum(wts$w1), sum(wts$w2), sum(wts$w3)) / N

    # Keep the iteration off the boundary of S^3: the four vertices are fixed
    # points of Phi (Supplemental A.2) and log(0) would poison later terms.
    kappa_new <- pmax(kappa_new, floor_kappa)
    kappa_new <- kappa_new / sum(kappa_new)

    if (max(abs(kappa_new - kappa)) < tol) {
      return(kappa_new)
    }
    kappa <- kappa_new
  }

  kappa
}
