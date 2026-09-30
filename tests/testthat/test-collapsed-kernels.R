# Collapsed kernel layer: log marginal likelihoods, posterior predictives and
# sufficient-statistic bookkeeping (spec tests T1-T3)

Y_test <- make_test_response()

for (cfg in names(kernel_configs)) {
  test_that(paste0("T1 [", cfg, "]: predictive = ratio of marginals"), {
    resp <- prep_response(Y_test[, kernel_configs[[cfg]], drop = FALSE])
    set.seed(11)
    expect_identical(cm_log_m(resp, integer(0)), 0)
    for (nD in c(0, 1, 5, 40)) {
      i <- sample(resp$n, 1)
      idx <- sample(setdiff(seq_len(resp$n), i), nD)
      ratio <- cm_log_m(resp, c(idx, i)) - cm_log_m(resp, idx)
      # closed form (R), sampler path with i outside the cluster, and
      # sampler path with i inside the cluster (leave-one-out)
      expect_close(ref_pred(resp, idx, i), ratio)
      expect_close(cm_log_pred(resp, idx, i), ratio)
      expect_close(cm_log_pred(resp, c(idx, i), i), ratio)
    }
  })
}

test_that("T2: Gaussian predictive matches mvtnorm::dmvt", {
  skip_if_not_installed("mvtnorm")
  set.seed(12)
  for (cols in list(kernel_configs$gauss3, kernel_configs$gauss1)) {
    resp <- prep_response(Y_test[, cols, drop = FALSE])
    pp <- resp$pp
    q <- ncol(resp$Y_N)
    for (nD in c(0, 1, 5, 40)) {
      i <- sample(resp$n, 1)
      idx <- sample(setdiff(seq_len(resp$n), i), nD)
      po <- niw_post(resp$Y_N[idx, , drop = FALSE], pp$beta0, pp$nu0, pp$eps0)
      df <- po$nun - q + 1
      ref <- mvtnorm::dmvt(
        resp$Y_N[i, ],
        delta = po$mn,
        sigma = (po$bn + 1) / (po$bn * df) * po$Psin,
        df = df,
        log = TRUE
      )
      expect_close(cm_log_pred(resp, idx, i), ref)
      expect_close(ref_pred(resp, idx, i), ref)
    }
  }
})

test_that("T2: Poisson predictive matches dnbinom", {
  set.seed(13)
  resp <- prep_response(Y_test[, kernel_configs$pois, drop = FALSE])
  a <- resp$pp$alpha_p
  b <- resp$pp$beta_p
  for (nD in c(0, 1, 5, 40)) {
    i <- sample(resp$n, 1)
    idx <- sample(setdiff(seq_len(resp$n), i), nD)
    S <- colSums(resp$Y_P[idx, , drop = FALSE])
    ref <- sum(stats::dnbinom(
      resp$Y_P[i, ],
      size = a + S,
      prob = (b + nD) / (b + nD + 1),
      log = TRUE
    ))
    expect_close(cm_log_pred(resp, idx, i), ref)
  }
})

for (cfg in names(kernel_configs)) {
  test_that(paste0("T3 [", cfg, "]: chain rule and order invariance"), {
    resp <- prep_response(Y_test[, kernel_configs[[cfg]], drop = FALSE])
    set.seed(14)
    D <- sample(resp$n, 30)
    lm <- cm_log_m(resp, D)
    for (rep in 1:2) {
      perm <- sample(D)
      # log m(D) = sum_j log p(y_j | y_1, ..., y_{j-1}), via both paths
      chain_out <- sum(vapply(
        seq_along(perm),
        \(j) cm_log_pred(resp, perm[seq_len(j - 1)], perm[j]),
        numeric(1)
      ))
      chain_in <- sum(vapply(
        seq_along(perm),
        \(j) cm_log_pred(resp, perm[seq_len(j)], perm[j]),
        numeric(1)
      ))
      expect_close(chain_out, lm)
      expect_close(chain_in, lm)
      expect_close(cm_log_m(resp, perm), lm)
    }
    # Adding a superset and removing the extra points gives the same stats
    E <- sample(setdiff(seq_len(resp$n), D), 20)
    expect_close(cm_log_m(resp, sample(c(D, E)), remove = sample(E)), lm)
  })
}
