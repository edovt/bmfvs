# Posterior log odds (log_odds_h1), the ranking derived from it, and the
# empirical Bayes collapse to the kappa = (1, 0, 0, 0) corner.

# S x p log Bayes factor matrices with the given column means
lbf <- function(S, means, sd) {
  matrix(rnorm(S * length(means), rep(means, each = S), sd), S)
}

log_mean_exp <- function(v) max(v) + log(mean(exp(v - max(v))))

test_that("log_odds_h1: C++ matches R and equals logit(1 - prob_null)", {
  set.seed(1)
  B1 <- lbf(60, rnorm(80, -2, 4), 1.5)
  B2 <- lbf(60, rnorm(80, -3, 4), 1.5)
  kap <- estimate_kappa(B1, B2)
  pc <- posterior_probs(B1, B2, kap, backend = "cpp")
  pr <- posterior_probs(B1, B2, kap, backend = "r")
  expect_identical(pc$prob_null, pr$prob_null)
  expect_equal(pc$log_odds_h1, pr$log_odds_h1, tolerance = 1e-12)
  ok <- pc$prob_null > 1e-10 & pc$prob_null < 1 - 1e-10
  expect_gt(sum(ok), 40)
  expect_equal(pc$log_odds_h1[ok], qlogis(1 - pc$prob_null[ok]), tolerance = 1e-8)
})

test_that("ranking by log_odds_h1 equals ranking by prob_null without ties", {
  set.seed(2)
  B1 <- lbf(50, rnorm(200, -1, 3), 1)
  B2 <- lbf(50, rnorm(200, -2, 3), 1)
  kap <- estimate_kappa(B1, B2)
  pc <- posterior_probs(B1, B2, kap)
  expect_false(anyDuplicated(pc$prob_null) > 0)
  expect_identical(rank(-pc$log_odds_h1, ties.method = "min"),
                   rank(pc$prob_null, ties.method = "min"))
})

test_that("at the kappa corner prob_null rounds to 1 but log_odds_h1 ranks", {
  set.seed(3)
  S <- 40; p <- 300
  # every Bayes factor < 1; the first 5 predictors are less negative
  B1 <- lbf(S, c(rep(-35, 5), rep(-60, p - 5)), 3)
  B2 <- lbf(S, rep(-200, p), 5)
  kap <- estimate_kappa(B1, B2)
  expect_true(eb_corner(log_mean_bf(B1, B2)))
  pc <- posterior_probs(B1, B2, kap, backend = "cpp")
  pr <- posterior_probs(B1, B2, kap, backend = "r")
  expect_true(all(pc$prob_null == 1))                 # the rounding problem
  expect_true(all(is.finite(pc$log_odds_h1)))
  expect_equal(pc$log_odds_h1, pr$log_odds_h1, tolerance = 1e-12)
  expect_equal(anyDuplicated(pc$log_odds_h1), 0L)
  expect_setequal(order(-pc$log_odds_h1)[1:5], 1:5)
  # near the corner: log odds = log P(H1) - log P(H0) with log P(H0) ~ 0 and
  # P(H1) ~ mean_s sum_t kappa_t BF_jt / kappa_0
  lk <- log(kap)
  approx <- sapply(seq_len(p), \(j) {
    a <- cbind(lk[2] + B1[, j], lk[3] + B2[, j], lk[4] + B1[, j] + B2[, j])
    log_mean_exp(apply(a, 1, \(r) max(r) + log(sum(exp(r - max(r)))))) - lk[1]
  })
  expect_equal(pc$log_odds_h1, approx, tolerance = 1e-8)
})

test_that("log_odds_h1 also separates predictors whose prob_null is 0", {
  set.seed(4)
  B1 <- lbf(40, c(800, 900, 1000, rep(-5, 50)), 2)
  B2 <- lbf(40, rep(-5, 53), 2)
  kap <- estimate_kappa(B1, B2)
  pc <- posterior_probs(B1, B2, kap)
  expect_true(all(pc$prob_null[1:3] == 0))
  expect_equal(anyDuplicated(pc$log_odds_h1[1:3]), 0L)
  expect_identical(order(-pc$log_odds_h1)[1:3], order(pc$log_prob_null)[1:3])
})

test_that("log_mean_bf(): C++ matches R; corner iff all mean BFs < 1", {
  set.seed(8)
  B1 <- lbf(30, rnorm(40, -3, 2), 1)
  B2 <- lbf(30, rnorm(40, -6, 2), 1)
  a <- log_mean_bf(B1, B2, backend = "cpp")
  b <- log_mean_bf(B1, B2, backend = "r")
  expect_equal(a, b, tolerance = 1e-12)
  expect_equal(unname(a[1]), log(mean(exp(B1))), tolerance = 1e-12)
  expect_true(eb_corner(a))
  # one strong predictor (BF ~ e^10 > p) puts the mean above 1
  B1[, 1] <- B1[, 1] + 15
  expect_false(eb_corner(log_mean_bf(B1, B2)))
  # and empirical Bayes then keeps kappa_1 in the interior
  expect_gt(estimate_kappa(B1, B2)[2], 1e-3)
})

test_that("bmfvs(): rank from log_odds_h1, warning only at the corner", {
  set.seed(5)
  n <- 150; p <- 30
  X <- matrix(rbinom(n * p, 1, 0.5), n, p)
  # signal: no collapse, no warning, rank = rank(prob_null) when untied
  y <- rnorm(n, mean = 6 * X[, 1])
  set.seed(6)
  expect_no_warning(
    fit <- bmfvs(y, X, n_iter = 200, n_burnin = 100, k = 2, y_types = "g",
                 verbose = FALSE, n_threads = 1)
  )
  expect_false(fit$eb_collapsed)
  expect_identical(unname(fit$rank), unname(rank(-fit$log_odds_h1, ties.method = "min")))
  if (!anyDuplicated(fit$prob_null)) {
    expect_identical(unname(fit$rank), rank(unname(fit$prob_null), ties.method = "min"))
  }
  expect_identical(unname(which(fit$rank == 1)), 1L)
  # pure noise: Bayes factors < 1 on average -> corner, warning, still a ranking
  y0 <- rnorm(n)
  set.seed(7)
  expect_warning(
    fit0 <- bmfvs(y0, X, n_iter = 200, n_burnin = 100, k = 2, y_types = "g",
                  verbose = FALSE, n_threads = 1),
    "no predictor is individually supported"
  )
  expect_true(fit0$eb_collapsed)
  expect_true(all(fit0$prob_null > 0.999))
  expect_true(all(is.finite(fit0$log_odds_h1)))
  expect_setequal(unname(fit0$rank), seq_len(p))
  # empirical_bayes = FALSE never warns
  set.seed(7)
  expect_no_warning(
    bmfvs(y0, X, n_iter = 200, n_burnin = 100, k = 2, y_types = "g",
          verbose = FALSE, n_threads = 1, empirical_bayes = FALSE)
  )
})
