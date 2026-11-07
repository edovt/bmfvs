# Both backends of compute_log_bfs() against the plain-R transcription of the
# paper formulas (helper-reference-bf.R), for every kernel

bf_case <- function(qN, m, qP, n = 30, k = 3, p = 6, S = 2, seed = 1) {
  set.seed(seed)
  X <- sapply(1:p, \(j) sample(0:(sample(1:3, 1)), n, TRUE))
  X[, 1] <- 0L; X[, 2] <- sample(c(0L, 2L), n, TRUE)  # constant; empty level
  spd <- function(q) { A <- matrix(rnorm(q * q), q); crossprod(A) / q + diag(.3, q) }
  rd <- function(a) { g <- rgamma(length(a), a); g / sum(g) }
  fit <- list(z = matrix(sample.int(k, S * n, TRUE), S, n), w = t(replicate(S, rd(rep(1, k)))))
  fit$z[1, ] <- sample.int(k - 1, n, TRUE)             # an empty component
  Y_N <- Y_C <- Y_P <- NULL
  if (qN) {
    Y_N <- matrix(rnorm(n * qN), n)
    fit$mu <- array(rnorm(S * k * qN), c(S, k, qN))
    fit$Sigma <- array(0, c(S, k, qN, qN))
    for (s in 1:S) for (h in 1:k) fit$Sigma[s, h, , ] <- spd(qN)
  }
  if (length(m)) {
    Y_C <- matrix(as.integer(sapply(m, \(mm) sample.int(mm, n, TRUE))), n)
    fit$psi <- lapply(m, \(mm) { a <- array(0, c(S, k, mm))
      for (s in 1:S) for (h in 1:k) a[s, h, ] <- rd(rep(1, mm)); a })
  }
  if (qP) {
    Y_P <- matrix(as.numeric(rpois(n * qP, 4)), n)
    fit$rho <- array(rgamma(S * k * qP, 4), c(S, k, qP))
  }
  list(X = X, Y_N = Y_N, Y_C = Y_C, Y_P = Y_P, fit = fit, k = k, S = S, p = p,
       tw = 3.7, tmu = 1.3, tS = qN + 4.2, tpsi = 6.1, trho = 2.4)
}

bf_reference <- function(cs) {
  f <- cs$fit; k <- cs$k
  B1 <- B2 <- matrix(0, cs$S, cs$p)
  for (s in 1:cs$S) for (j in 1:cs$p) {
    z <- f$z[s, ]; xj <- cs$X[, j]
    B1[s, j] <- ref_bf1(z, f$w[s, ], xj, cs$tw, k)
    if (!is.null(cs$Y_N)) {
      q <- ncol(cs$Y_N)
      B2[s, j] <- B2[s, j] + ref_bf2_gauss(z, cs$Y_N, xj, lapply(1:k, \(h) f$mu[s, h, ]),
        lapply(1:k, \(h) matrix(f$Sigma[s, h, , ], q)), cs$tmu, cs$tS, k)
    }
    if (!is.null(cs$Y_C)) {
      B2[s, j] <- B2[s, j] + ref_bf2_cat(z, cs$Y_C, xj, lapply(f$psi, \(a) matrix(a[s, , ], k)), cs$tpsi, k)
    }
    if (!is.null(cs$Y_P)) {
      B2[s, j] <- B2[s, j] + ref_bf2_pois(z, cs$Y_P, xj, matrix(f$rho[s, , ], k), cs$trho, k)
    }
  }
  list(log_bf1 = B1, log_bf2 = B2)
}

bf_configs <- list(
  gauss1 = list(1, integer(0), 0), gauss3 = list(3, integer(0), 0),
  cat = list(0, c(2L, 4L), 0), pois = list(0, integer(0), 2),
  mixed = list(2, 3L, 1)
)
for (cfg in names(bf_configs)) {
  test_that(paste0("log BFs match the paper formulas [", cfg, "]"), {
    a <- bf_configs[[cfg]]
    cs <- bf_case(a[[1]], a[[2]], a[[3]])
    ref <- bf_reference(cs)
    for (be in c("cpp", "r")) {
      got <- compute_log_bfs(cs$fit, cs$Y_N, cs$Y_C, cs$Y_P, cs$X, cs$tw,
        tau_mu = if (!is.null(cs$Y_N)) cs$tmu, tau_Sigma = if (!is.null(cs$Y_N)) cs$tS,
        tau_psi = if (!is.null(cs$Y_C)) cs$tpsi, tau_rho = if (!is.null(cs$Y_P)) cs$trho,
        backend = be, n_threads = 1)
      expect_equal(got$log_bf1, ref$log_bf1, tolerance = 1e-10)
      expect_equal(got$log_bf2, ref$log_bf2, tolerance = 1e-10)
    }
  })
}
