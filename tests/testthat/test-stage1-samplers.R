# Stage-1 samplers through fit_gibbs_mixed(): output contract, prior
# recovery and agreement with the conditional sampler (spec tests T7-T8)

samplers <- c("nonreversible", "marginal", "conditional")

# Dimensions and storage modes of every element of a fit
fit_shape <- function(x) {
  if (is.list(x)) {
    return(lapply(x, fit_shape))
  }
  list(dim = dim(x), mode = storage.mode(x))
}

test_that("all samplers return the conditional sampler's output format", {
  Y <- make_test_response(n = 40)
  for (cfg in c("gauss3", "cat", "pois", "mixed")) {
    resp <- prep_response(Y[, kernel_configs[[cfg]], drop = FALSE])
    fits <- lapply(setNames(nm = samplers), \(s) {
      set.seed(31)
      fit_gibbs_mixed(resp$Y_N, resp$Y_C, resp$Y_P, 3L, 30L, 10L, NULL, resp$pp, s)
    })
    for (s in samplers) {
      f <- fits[[s]]
      expect_identical(attr(f, "sampler"), s)
      expect_identical(fit_shape(unclass(f)), fit_shape(unclass(fits$conditional)))
      expect_true(all(f$z %in% 1:3))
      expect_equal(rowSums(f$w), rep(1, 20))
      expect_true(all(is.finite(f$log_lik)))
    }
  }
})

# T7 ------------------------------------------------------------------------

# Draw the response blocks of n observations from the prior predictive of the
# mixture (k components, alpha/k = a_k, default priors). Datasets missing the
# top level of a categorical response are redrawn, since the fitted model
# reads the number of levels from the data.
sim_prior_blocks <- function(n, k, a_k, qN, m, qP) {
  pp <- expand_prior_params(list(), qN, length(m), qP)
  repeat {
    w <- stats::rgamma(k, a_k)
    z <- sample.int(k, n, TRUE, prob = w / sum(w))
    Y_N <- matrix(0, n, qN)
    Y_C <- matrix(0L, n, length(m))
    Y_P <- matrix(0, n, qP)
    for (h in seq_len(k)) {
      idx <- which(z == h)
      nh <- length(idx)
      if (qN > 0) {
        # Sigma ~ IW(nu0, eps0 I), mu | Sigma ~ N(0, Sigma / beta0)
        Sigma <- solve(stats::rWishart(1, pp$nu0, diag(qN) / pp$eps0)[, , 1])
        R <- chol(Sigma)
        mu <- drop(crossprod(R, stats::rnorm(qN))) / sqrt(pp$beta0)
        Y_N[idx, ] <- sweep(matrix(stats::rnorm(nh * qN), nh, qN) %*% R, 2, mu, "+")
      }
      for (t in seq_along(m)) {
        # psi ~ Dir(alpha_t / m_t, ...) with the default alpha_t = m_t
        psi <- stats::rgamma(m[t], 1)
        Y_C[idx, t] <- sample.int(m[t], nh, TRUE, prob = psi / sum(psi))
      }
      for (t in seq_len(qP)) {
        rho <- stats::rgamma(1, pp$alpha_p[t], pp$beta_p[t])
        Y_P[idx, t] <- stats::rpois(nh, rho)
      }
    }
    if (all(apply(Y_C, 2, max) == m)) break
  }
  list(Y_N = Y_N, Y_C = Y_C, Y_P = Y_P, pp = pp)
}

# CDF of n_1 ~ BetaBinomial(n, a, b), made continuous by spreading each atom
# uniformly over [x, x + 1), so the KS test applies to n_1 + Unif(0, 1)
pbetabinom_cont <- function(x, n, a, b) {
  pmf <- exp(lchoose(n, 0:n) + lbeta(0:n + a, n - 0:n + b) - lbeta(a, b))
  cdf <- c(0, cumsum(pmf))
  f <- pmin(floor(x), n)
  cdf[f + 1] + (x - f) * pmf[f + 1]
}

test_that("T7: samplers started from uniform z recover the prior of n_1", {
  skip_if_not_slow()
  n <- 200
  k <- 3
  a_k <- 1
  R <- 200
  settings <- list(
    gauss = list(qN = 2, m = integer(0), qP = 0),
    cat = list(qN = 0, m = c(3L, 2L), qP = 0),
    pois = list(qN = 0, m = integer(0), qP = 2),
    mixed = list(qN = 1, m = 3L, qP = 1)
  )
  for (nm in names(settings)) {
    set.seed(71)
    sims <- replicate(
      R,
      do.call(sim_prior_blocks, c(list(n = n, k = k, a_k = a_k), settings[[nm]])),
      simplify = FALSE
    )
    for (s in c("nonreversible", "marginal")) {
      # size of component 1 after 100 sweeps from a uniform allocation
      n1 <- vapply(
        sims,
        \(d) {
          fit <- fit_gibbs_mixed(d$Y_N, d$Y_C, d$Y_P, k, 100L, 99L, a_k * k, d$pp, s)
          sum(fit$z[1, ] == 1)
        },
        numeric(1)
      )
      p <- stats::ks.test(
        n1 + stats::runif(R),
        pbetabinom_cont,
        n = n,
        a = a_k,
        b = (k - 1) * a_k
      )$p.value
      expect_gt(p, 0.01, label = sprintf("KS p-value [%s, %s]", nm, s))
    }
  }
})

# T8 ------------------------------------------------------------------------

# Posterior co-clustering matrix P(z_i = z_j | Y)
co_clustering <- function(z, k) {
  C <- matrix(0, ncol(z), ncol(z))
  for (h in seq_len(k)) {
    I <- (z == h) * 1
    C <- C + crossprod(I)
  }
  C / nrow(z)
}

# Posterior mean marginal density of each response on a grid
marginal_densities <- function(fit, resp) {
  k <- ncol(fit$w)
  out <- list()
  for (d in seq_len(ncol(resp$Y_N))) {
    x <- seq(min(resp$Y_N[, d]) - 1, max(resp$Y_N[, d]) + 1, length.out = 60)
    f <- 0
    for (h in seq_len(k)) {
      sd_h <- sqrt(fit$Sigma[, h, d, d])
      f <- f + colMeans(fit$w[, h] * sapply(x, \(xx) stats::dnorm(xx, fit$mu[, h, d], sd_h)))
    }
    out[[paste0("N", d)]] <- f
  }
  for (t in seq_along(fit$psi)) {
    m_t <- dim(fit$psi[[t]])[3]
    out[[paste0("C", t)]] <- sapply(seq_len(m_t), \(l) mean(rowSums(fit$w * fit$psi[[t]][, , l])))
  }
  for (t in seq_len(ncol(resp$Y_P))) {
    y <- 0:(max(resp$Y_P[, t]) + 5)
    f <- 0
    for (h in seq_len(k)) {
      f <- f + colMeans(fit$w[, h] * sapply(y, \(yy) stats::dpois(yy, fit$rho[, h, t])))
    }
    out[[paste0("P", t)]] <- f
  }
  out
}

test_that("T8: collapsed and conditional samplers agree on posterior summaries", {
  skip_if_not_slow()
  # Three moderately overlapping groups, so that many co-clustering
  # probabilities are far from 0 and 1
  set.seed(8)
  n <- 90
  g <- rep(1:3, each = 30)
  probs <- list(c(.7, .2, .1), c(.2, .6, .2), c(.1, .2, .7))
  Y <- data.frame(
    y1 = stats::rnorm(n, c(-1.5, 1.5, 0)[g]),
    y2 = stats::rnorm(n, c(0, 0, 2)[g]),
    c1 = factor(sapply(g, \(h) sample(1:3, 1, prob = probs[[h]]))),
    p1 = as.integer(stats::rpois(n, c(3, 6, 12)[g]))
  )
  resp <- prep_response(Y)
  k <- 3L
  summ <- lapply(setNames(nm = samplers), \(s) {
    set.seed(100)
    fit <- fit_gibbs_mixed(resp$Y_N, resp$Y_C, resp$Y_P, k, 6000L, 1000L, NULL, resp$pp, s)
    list(C = co_clustering(fit$z, k), D = marginal_densities(fit, resp))
  })
  # Tolerances are about 2-3 times the differences between two seeds of the
  # same sampler on this dataset (co-clustering: max 0.05, mean 0.006;
  # densities: 0.8% of the peak)
  for (s in c("nonreversible", "marginal")) {
    a <- summ[[s]]
    b <- summ$conditional
    dC <- abs(a$C - b$C)
    dD <- mapply(\(x, y) max(abs(x - y)) / max(y), a$D, b$D)
    expect_lt(max(dC), 0.1, label = paste("max co-clustering difference,", s))
    expect_lt(mean(dC), 0.015, label = paste("mean co-clustering difference,", s))
    expect_lt(max(dD), 0.03, label = paste("max relative density difference,", s))
  }
})
