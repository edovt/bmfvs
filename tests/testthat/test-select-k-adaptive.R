# Adaptive WAIC search for k (k = NULL in select_k() and bmfvs())

# n observations from m well-separated univariate Gaussian clusters
sep_clusters <- function(n, m, seed = 1) {
  set.seed(seed)
  rnorm(n, mean = 12 * sample.int(m, n, replace = TRUE), sd = 0.5)
}

# loo_compare() warns that a best-of-many choice may be due to chance
# (McLatchie and Vehtari, 2023); expected here, so muffle only that message.
quiet_loo <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    if (grepl("McLatchie", conditionMessage(w))) invokeRestart("muffleWarning")
  })
}

elpds <- function(sel) {
  cmp <- sel$comparison
  e <- cmp[, "elpd_waic"]
  names(e) <- if (is.data.frame(cmp) && !is.null(cmp$model)) cmp$model else rownames(cmp)
  e[order(as.integer(sub("k=", "", names(e))))]
}

test_that("first window of the adaptive search reproduces k = 1:k_init", {
  y <- sep_clusters(120, 2, seed = 11)
  set.seed(2)
  fixed <- select_k(y, k = 1:6, n_iter = 200, n_burnin = 100,
                    y_types = "gaussian", verbose = FALSE)
  set.seed(2)
  adapt <- select_k(y, k = NULL, n_iter = 200, n_burnin = 100,
                    y_types = "gaussian", verbose = FALSE,
                    k_init = 6, k_step = 2, k_max = 12)
  expect_identical(fixed$k_candidates, 1:6)
  expect_identical(adapt$k_candidates[1:6], 1:6)
  expect_identical(elpds(adapt)[1:6], elpds(fixed))
  if (identical(adapt$k_candidates, 1:6)) {
    expect_identical(adapt$k, fixed$k)
    expect_identical(adapt$fit$z, fixed$fit$z)
  }
})

test_that("the search extends while the best k is in the top window", {
  y <- sep_clusters(300, 8, seed = 3)
  set.seed(4)
  sel <- quiet_loo(select_k(y, k = NULL, n_iter = 300, n_burnin = 150,
                            y_types = "gaussian", verbose = FALSE,
                            k_init = 3, k_step = 2, k_max = 30))
  kc <- sel$k_candidates
  upper <- max(kc)
  # contiguous, each k fitted once
  expect_identical(kc, seq_len(upper))
  expect_identical(length(elpds(sel)), upper)
  # it went beyond the initial window and found the 8 clusters (roughly)
  expect_gt(upper, 3)
  expect_gte(sel$k, 6)
  # stopping rule: best k is below the top window, or the cap was reached
  expect_true(sel$k < upper - 2 || upper == 30)
  # the selected k is the WAIC maximum over all fitted k
  e <- elpds(sel)
  expect_identical(sel$k, as.integer(which.max(e)))
})

test_that("reaching k_max with the best k in the top window warns", {
  y <- sep_clusters(300, 8, seed = 3)
  set.seed(4)
  expect_warning(
    sel <- select_k(y, k = NULL, n_iter = 300, n_burnin = 150,
                    y_types = "gaussian", verbose = FALSE,
                    k_init = 2, k_step = 2, k_max = 4),
    "k_max"
  )
  expect_identical(sel$k_candidates, 1:4)
})

test_that("numeric k is unaffected by the search settings", {
  y <- sep_clusters(100, 2, seed = 5)
  set.seed(6)
  a <- select_k(y, k = 1:4, n_iter = 150, n_burnin = 50,
                y_types = "gaussian", verbose = FALSE)
  set.seed(6)
  b <- select_k(y, k = 1:4, n_iter = 150, n_burnin = 50,
                y_types = "gaussian", verbose = FALSE,
                k_init = 1, k_step = 1, k_max = 2)
  expect_identical(a$k, b$k)
  expect_identical(a$fit$z, b$fit$z)
  expect_identical(b$k_candidates, 1:4)
})

test_that("invalid search settings are rejected", {
  y <- sep_clusters(50, 2)
  for (bad in list(list(k_init = 0), list(k_step = 1.5), list(k_max = -1))) {
    expect_error(
      do.call(select_k, c(list(y, k = NULL, n_iter = 20, n_burnin = 10,
                               y_types = "gaussian", verbose = FALSE), bad)),
      "must be a single integer"
    )
  }
})

test_that("bmfvs() runs with the default k = NULL", {
  set.seed(7)
  n <- 150
  X <- matrix(rbinom(n * 20, 1, 0.5), n, 20)
  y <- rnorm(n, mean = 6 * X[, 1])
  fit <- bmfvs(y, X, n_iter = 200, n_burnin = 100, y_types = "gaussian",
               verbose = FALSE, n_threads = 1, k_init = 4, k_step = 2,
               k_max = 10)
  expect_true(fit$k %in% fit$k_candidates)
  expect_identical(fit$k_candidates, seq_len(max(fit$k_candidates)))
  expect_length(fit$prob_null, 20)
  expect_lt(fit$prob_null[1], 0.5)
})
