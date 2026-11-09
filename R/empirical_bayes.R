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

# Lower bound on each kappa_t during the fixed point iteration (keeps the
# iterates off the boundary of S^3; see estimate_kappa()). Also used by
# kappa_collapsed() to detect convergence to the kappa = (1, 0, 0, 0) corner.
kappa_floor <- 1e-12

# log of the Bayes factors BF_j1, BF_j2, BF_j3 averaged over all predictors and
# posterior samples. Near the kappa = (1, 0, 0, 0) corner the fixed point map
# behaves like Phi_t(kappa) ~ kappa_t * mean(BF_jt), so the corner attracts the
# empirical Bayes iteration exactly when all three are < 0. That condition is
# what eb_corner() checks; it does not depend on where the iteration happened
# to stop (with weak evidence kappa_t decays slowly and the tolerance stops it
# well above the floor).
log_mean_bf <- function(log_bf1, log_bf2, backend = c("cpp", "r")) {
  backend <- match.arg(backend)
  if (!is.double(log_bf1)) storage.mode(log_bf1) <- "double"
  if (!is.double(log_bf2)) storage.mode(log_bf2) <- "double"
  if (backend == "cpp") {
    return(log_mean_bf_cpp(log_bf1, log_bf2))
  }
  lme <- function(v) max(v) + log(mean(exp(v - max(v))))
  b1 <- as.vector(log_bf1)
  b2 <- as.vector(log_bf2)
  c(H1_weights = lme(b1), H1_kernel = lme(b2), H1_both = lme(b1 + b2))
}

eb_corner <- function(log_mean_bf) all(log_mean_bf < 0)

# Empirical Bayes estimation of the prior probabilities kappa (2.9), as the
# fixed point of Phi in (2.10).
#
# N * Phi(kappa) is evaluated by eb_phi_sums_cpp() in a single pass over the
# S x p log Bayes factor matrices, with the same per-element operations and
# long double accumulation as sum(bf_weights(...)$w*), so the iterates are
# bitwise identical to the R reference (backend = "r") without its ~12
# length-S*p temporaries per iteration.
estimate_kappa <- function(
  log_bf1,
  log_bf2,
  kappa_init = c(1 / 2, 1 / 6, 1 / 6, 1 / 6),
  max_iter = 100,
  tol = 1e-4,
  floor_kappa = kappa_floor,
  backend = c("cpp", "r")
) {
  backend <- match.arg(backend)
  N <- length(log_bf1)
  if (length(log_bf2) != N) {
    stop("log_bf1 and log_bf2 must have the same dimensions")
  }
  if (!is.double(log_bf1)) storage.mode(log_bf1) <- "double"
  if (!is.double(log_bf2)) storage.mode(log_bf2) <- "double"

  if (backend == "cpp") {
    phi_sums <- function(log_kappa) eb_phi_sums_cpp(log_bf1, log_bf2, log_kappa)
  } else {
    # Flattened once: the fixed point iteration only changes kappa
    b1 <- as.vector(log_bf1)
    b2 <- as.vector(log_bf2)
    b12 <- b1 + b2
    phi_sums <- function(log_kappa) {
      wts <- bf_weights(log_kappa, b1, b2, b12)
      c(sum(wts$w0), sum(wts$w1), sum(wts$w2), sum(wts$w3))
    }
  }

  kappa <- kappa_init

  for (iter in seq_len(max_iter)) {
    kappa_new <- phi_sums(log(kappa)) / N

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

# Posterior probabilities (2.7)-(2.8) given kappa: Monte Carlo averages over
# the S posterior samples (rows) of the normalized weights, per predictor.
#   prob_null      P(H0j | Y, X_j), identical to
#                  colMeans(matrix(bf_weights(...)$w0, nrow = S));
#   prob_h1        p x 3 matrix of P(H1jt | Y, X_j), t = 1, 2, 3;
#   log_prob_null  log P(H0j | Y, X_j) by log-sum-exp, finite when prob_null
#                  underflows to 0 (strong signals), so ties at 0 are broken;
#   log_odds_h1    log P(H1j | Y, X_j) - log P(H0j | Y, X_j) = logit(1 -
#                  prob_null), both terms by log-sum-exp, so it stays finite
#                  and ordered when prob_null rounds to 1 (empirical Bayes at
#                  the kappa = (1, 0, 0, 0) corner) as well as to 0.
posterior_probs <- function(log_bf1, log_bf2, kappa, backend = c("cpp", "r")) {
  backend <- match.arg(backend)
  log_bf1 <- as.matrix(log_bf1)
  log_bf2 <- as.matrix(log_bf2)
  if (!is.double(log_bf1)) storage.mode(log_bf1) <- "double"
  if (!is.double(log_bf2)) storage.mode(log_bf2) <- "double"
  if (backend == "cpp") {
    out <- posterior_probs_cpp(log_bf1, log_bf2, log(kappa))
  } else {
    S <- nrow(log_bf1)
    b1 <- as.vector(log_bf1)
    b2 <- as.vector(log_bf2)
    wts <- bf_weights(log(kappa), b1, b2, b1 + b2)
    lk <- log(kappa)
    t <- cbind(lk[1], lk[2] + b1, lk[3] + b2, lk[4] + (b1 + b2))
    m <- do.call(pmax.int, lapply(1:4, \(i) t[, i]))
    ld <- log(rowSums(exp(t - m)))
    lw0 <- matrix((lk[1] - m) - ld, nrow = S)
    ma <- do.call(pmax.int, lapply(2:4, \(i) t[, i]))
    lw1 <- matrix((ma - m) + log(rowSums(exp(t[, 2:4] - ma))) - ld, nrow = S)
    log_mean_exp <- function(v) {
      mx <- max(v)
      if (!is.finite(mx)) return(mx)
      mx + log(sum(exp(v - mx))) - log(S)
    }
    out <- list(
      prob_null = colMeans(matrix(wts$w0, nrow = S)),
      prob_h1 = cbind(
        colMeans(matrix(wts$w1, nrow = S)),
        colMeans(matrix(wts$w2, nrow = S)),
        colMeans(matrix(wts$w3, nrow = S))
      ),
      log_prob_null = apply(lw0, 2, \(v) {
        mx <- max(v)
        mx + log(sum(exp(v - mx))) - log(S)
      })
    )
    out$log_odds_h1 <- apply(lw1, 2, log_mean_exp) - out$log_prob_null
  }
  colnames(out$prob_h1) <- c("H1_weights", "H1_kernel", "H1_both")
  out
}
