# Empirical Bayes fixed point (2.10) and posterior probabilities (2.7)-(2.8)

make_log_bfs <- function(S = 40, p = 120, seed = 1) {
  set.seed(seed)
  B1 <- matrix(rnorm(S * p, -3, 1), S, p)
  B2 <- matrix(rnorm(S * p, -2, 1.5), S, p)
  B1[, 1:6] <- B1[, 1:6] + 9
  B2[, 7:12] <- B2[, 7:12] + 30
  # strong signals: the posterior mean of w0 underflows to exactly 0
  B1[, 13:15] <- B1[, 13:15] + c(760, 800, 900)[col(B1[, 13:15])]
  B2[, 16] <- B2[, 16] + 1000
  list(B1 = B1, B2 = B2)
}

test_that("C++ fixed point is bitwise identical to the R reference", {
  for (seed in 1:3) {
    b <- make_log_bfs(seed = seed)
    for (init in list(c(1 / 2, 1 / 6, 1 / 6, 1 / 6), c(.25, .25, .25, .25))) {
      k_cpp <- estimate_kappa(b$B1, b$B2, kappa_init = init, backend = "cpp")
      k_r <- estimate_kappa(b$B1, b$B2, kappa_init = init, backend = "r")
      expect_identical(k_cpp, k_r)
    }
  }
})

test_that("estimate_kappa returns a fixed point of Phi to the tolerance", {
  b <- make_log_bfs()
  kap <- estimate_kappa(b$B1, b$B2, tol = 1e-10, max_iter = 10000)
  L <- cbind(log(kap[1]), log(kap[2]) + c(b$B1), log(kap[3]) + c(b$B2),
             log(kap[4]) + c(b$B1) + c(b$B2))
  W <- exp(L - apply(L, 1, max))
  phi <- colMeans(W / rowSums(W))
  expect_lt(max(abs(phi - kap)), 1e-9)
  expect_equal(sum(kap), 1)
})

test_that("posterior_probs: C++ matches R, log_prob_null resolves underflow", {
  b <- make_log_bfs()
  kap <- estimate_kappa(b$B1, b$B2)
  pc <- posterior_probs(b$B1, b$B2, kap, backend = "cpp")
  pr <- posterior_probs(b$B1, b$B2, kap, backend = "r")
  expect_identical(pc$prob_null, pr$prob_null)
  expect_identical(pc$prob_h1, pr$prob_h1)
  expect_equal(pc$log_prob_null, pr$log_prob_null, tolerance = 1e-12)

  # the four probabilities sum to one
  expect_lt(max(abs(pc$prob_null + rowSums(pc$prob_h1) - 1)), 1e-12)

  # log_prob_null is the log of the same posterior mean ...
  ok <- pc$prob_null > 1e-280
  expect_equal(exp(pc$log_prob_null[ok]), pc$prob_null[ok], tolerance = 1e-12)
  # ... and stays finite and ordered where prob_null underflows to 0
  zero <- which(pc$prob_null == 0)
  expect_true(length(zero) >= 2)
  expect_true(all(is.finite(pc$log_prob_null)))
  expect_equal(length(unique(pc$log_prob_null[zero])), length(zero))
  expect_true(all(pc$log_prob_null[zero] < log(.Machine$double.xmin)))
  # direct log-sum-exp of log w0 for one underflowing predictor
  j <- zero[1]
  t <- cbind(log(kap[1]), log(kap[2]) + b$B1[, j], log(kap[3]) + b$B2[, j],
             log(kap[4]) + b$B1[, j] + b$B2[, j])
  m <- apply(t, 1, max)
  lw0 <- (t[, 1] - m) - log(rowSums(exp(t - m)))
  ref <- max(lw0) + log(mean(exp(lw0 - max(lw0))))
  expect_equal(pc$log_prob_null[j], ref, tolerance = 1e-12)
})

test_that("bmfvs() returns log_prob_null and prob_h1 without changing prob_null", {
  set.seed(11)
  n <- 120; p <- 15
  X <- matrix(rbinom(n * p, 1, .5), n, p)
  y <- 6 * X[, 1] + rnorm(n) + (1 + X[, 2]) * rnorm(n)
  set.seed(12)
  res <- bmfvs(y, X, n_iter = 200, n_burnin = 100, k = 2, y_types = "g",
               verbose = FALSE, n_threads = 1)
  expect_named(res, c("prob_null", "kappa", "hyperparams", "log_bf1", "log_bf2",
                      "k", "waic", "fit", "y_types", "y_levels",
                      "log_prob_null", "prob_h1", "k_candidates",
                      "log_odds_h1", "rank", "eb_collapsed",
                      "eb_log_mean_bf"))
  expect_length(res$log_prob_null, p)
  expect_equal(dim(res$prob_h1), c(p, 3L))
  ref <- posterior_probs(res$log_bf1, res$log_bf2, res$kappa, backend = "r")
  expect_identical(unname(res$prob_null), ref$prob_null)
  ok <- res$prob_null > 1e-280
  expect_equal(unname(exp(res$log_prob_null[ok])), unname(res$prob_null[ok]),
               tolerance = 1e-12)
  expect_lt(max(abs(res$prob_null + rowSums(res$prob_h1) - 1)), 1e-12)
  expect_true(all(is.finite(res$log_prob_null)))
})
