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
  floor_kappa = 1e-12,
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
#                  underflows to 0 (strong signals), so ties at 0 are broken.
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
    lw0 <- matrix((lk[1] - m) - log(rowSums(exp(t - m))), nrow = S)
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
  }
  colnames(out$prob_h1) <- c("H1_weights", "H1_kernel", "H1_both")
  out
}
