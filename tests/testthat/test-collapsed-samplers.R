# Collapsed allocation samplers: cache bookkeeping, label invariance and
# exact invariance on enumerable problems (spec tests T4-T6)

Y_test <- make_test_response()

for (cfg in c("gauss3", "mixed")) {
  test_that(paste0("T4 [", cfg, "]: cached state matches recomputation"), {
    resp <- prep_response(Y_test[, kernel_configs[[cfg]], drop = FALSE])
    for (s in c("nonreversible", "marginal")) {
      set.seed(21)
      # 50 sweeps of n = 80 updates, with no resynchronisation of the cache
      chk <- collapsed_cache_check_cpp(
        resp$Y_N, resp$Y_C, resp$Y_P, resp$pp, 4L, 4, s, 50L
      )
      expect_identical(chk$int_err, 0L)
      expect_lt(chk$stat_err, 1e-8)
      expect_lt(chk$LN_err, 1e-8)
      expect_true(chk$members_ok)
      expect_gt(length(unique(chk$z)), 1)
    }
  })
}

test_that("T5: log collapsed posterior matches R and is label invariant", {
  set.seed(22)
  for (cfg in names(kernel_configs)) {
    resp <- prep_response(Y_test[, kernel_configs[[cfg]], drop = FALSE])
    for (a_k in c(1, 0.3)) {
      k <- 4
      z <- sample(k, resp$n, TRUE)
      lp <- collapsed_log_post_cpp(
        resp$Y_N, resp$Y_C, resp$Y_P, resp$pp, z, k, a_k * k
      )
      expect_close(lp, ref_log_post(resp, z, k, a_k * k))
      perm <- sample(k)
      lp_perm <- collapsed_log_post_cpp(
        resp$Y_N, resp$Y_C, resp$Y_P, resp$pp, perm[z], k, a_k * k
      )
      expect_close(lp_perm, lp)
    }
  }
})

# Tiny datasets (n = 6) whose k^n allocations can be enumerated
tiny_data <- list(
  pois = data.frame(y = as.integer(c(0, 1, 3, 7, 8, 12))),
  gauss = data.frame(
    y1 = c(-2.1, -1.7, -1.9, 1.6, 2.2, 1.4),
    y2 = c(0.3, -0.2, 0.1, 1.1, 0.6, 0.9)
  ),
  cat = data.frame(
    c1 = factor(c(1, 1, 1, 2, 2, 3)),
    c2 = factor(c(1, 2, 1, 2, 2, 2))
  ),
  mixed = data.frame(
    g = c(-1.1, -0.8, -1.3, 1.2, 0.9, 1.0),
    c = factor(c(1, 1, 2, 2, 2, 1)),
    p = as.integer(c(0, 2, 1, 7, 9, 6))
  )
)

# Exact pi(z | Y) over all k^n allocations, in expand.grid() row order
exact_alloc_post <- function(resp, k, alpha) {
  Z <- as.matrix(expand.grid(rep(list(seq_len(k)), resp$n)))
  lp <- apply(Z, 1, \(z) ref_log_post(resp, z, k, alpha))
  p <- exp(lp - max(lp))
  p / sum(p)
}

test_that("T6: samplers leave the exact collapsed posterior invariant", {
  skip_if_not_slow()
  set.seed(23)
  for (nm in names(tiny_data)) {
    resp <- prep_response(tiny_data[[nm]])
    for (k in 2:3) {
      for (a_k in c(1, 0.3, 2.5)) {
        p_exact <- exact_alloc_post(resp, k, a_k * k)
        for (s in c("nonreversible", "marginal")) {
          freq <- collapsed_state_freq_cpp(
            resp$Y_N, resp$Y_C, resp$Y_P, resp$pp, k, a_k * k, s, 1.5e6, 1e4
          )
          tv <- 0.5 * sum(abs(freq / sum(freq) - p_exact))
          expect_lt(tv, 0.05, label = sprintf("TV [%s, k=%d, a_k=%g, %s]", nm, k, a_k, s))
        }
      }
    }
  }
})
