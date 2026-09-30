# Shared fixtures for the collapsed-sampler tests

# Mixed-type response with every kind of block: a correlated, off-centre
# 3-d Gaussian, a 1-d Gaussian, categorical responses with 4, 10 and 2
# levels, and counts with small and large means
make_test_response <- function(n = 80, seed = 1) {
  set.seed(seed)
  L <- matrix(c(1, 0, 0, 0.6, 0.8, 0, -0.3, 0.2, 0.9), 3, 3)
  G <- matrix(rnorm(n * 3), n, 3) %*% t(L) * 2 + 1
  data.frame(
    g1 = G[, 1],
    g2 = G[, 2],
    g3 = G[, 3],
    h1 = rnorm(n, 5, 0.5),
    c1 = factor(sample(1:4, n, TRUE)),
    c2 = factor(sample(1:10, n, TRUE), levels = 1:10),
    c3 = factor(sample(1:2, n, TRUE)),
    p1 = as.integer(rpois(n, 5)),
    p2 = as.integer(rpois(n, 40))
  )
}

# Kernel configurations used across the tests (columns of the test response)
kernel_configs <- list(
  gauss3 = c("g1", "g2", "g3"),
  gauss1 = "h1",
  cat = c("c1", "c2", "c3"),
  pois = c("p1", "p2"),
  mixed = c("g1", "g2", "c1", "c3", "p1", "p2")
)

# Split the response into blocks and attach the default priors
prep_response <- function(Y) {
  resp <- split_response(Y)
  resp$pp <- expand_prior_params(
    list(),
    ncol(resp$Y_N),
    ncol(resp$Y_C),
    ncol(resp$Y_P)
  )
  resp$n <- nrow(Y)
  resp
}

cm_log_m <- function(resp, idx, remove = integer(0)) {
  collapsed_log_m_cpp(
    resp$Y_N,
    resp$Y_C,
    resp$Y_P,
    resp$pp,
    as.integer(idx),
    as.integer(remove)
  )
}

cm_log_pred <- function(resp, idx, i) {
  collapsed_log_pred_cpp(
    resp$Y_N,
    resp$Y_C,
    resp$Y_P,
    resp$pp,
    as.integer(idx),
    as.integer(i)
  )
}

# Reference closed forms (spec Section 3), written from the posterior
# parameters in the paper's scatter-matrix form rather than from the
# sufficient statistics the C++ code uses

niw_post <- function(D, beta0, nu0, eps0) {
  q <- ncol(D)
  n <- nrow(D)
  Psi0 <- eps0 * diag(q)
  if (n == 0) {
    return(list(bn = beta0, nun = nu0, mn = rep(0, q), Psin = Psi0))
  }
  ybar <- colMeans(D)
  C <- sweep(D, 2, ybar)
  bn <- beta0 + n
  list(
    bn = bn,
    nun = nu0 + n,
    mn = n * ybar / bn,
    Psin = Psi0 + crossprod(C) + beta0 * n / bn * tcrossprod(ybar)
  )
}

# Multivariate Student-t predictive of the NIW model
ref_pred_gauss <- function(y, D, beta0, nu0, eps0) {
  q <- length(y)
  po <- niw_post(D, beta0, nu0, eps0)
  d <- y - po$mn
  quad <- drop(crossprod(d, solve(po$Psin, d)))
  ld <- as.numeric(determinant(po$Psin)$modulus)
  r <- po$bn / (po$bn + 1)
  -q / 2 * log(pi) + lgamma((po$nun + 1) / 2) - lgamma((po$nun + 1 - q) / 2) +
    q / 2 * log(r) - ld / 2 - (po$nun + 1) / 2 * log1p(r * quad)
}

# prod_t (alpha_t/m_t + n_{t,y_t}) / (alpha_t + n)
ref_pred_cat <- function(y, D, alpha_c, m) {
  cnt <- vapply(seq_along(y), \(t) sum(D[, t] == y[t]), numeric(1))
  sum(log(alpha_c / m + cnt) - log(alpha_c + nrow(D)))
}

# prod_t NegBin(y_t | a_t + S_t, (b_t + n) / (b_t + n + 1)), written out
ref_pred_pois <- function(y, D, a, b) {
  n <- nrow(D)
  S <- colSums(D)
  sum(
    lgamma(a + S + y) -
      lgamma(a + S) -
      lgamma(y + 1) +
      (a + S) * log((b + n) / (b + n + 1)) -
      y * log(b + n + 1)
  )
}

# Sum of the block reference predictives of subject i given rows idx
ref_pred <- function(resp, idx, i) {
  pp <- resp$pp
  out <- 0
  if (ncol(resp$Y_N) > 0) {
    out <- out +
      ref_pred_gauss(
        resp$Y_N[i, ],
        resp$Y_N[idx, , drop = FALSE],
        pp$beta0,
        pp$nu0,
        pp$eps0
      )
  }
  if (ncol(resp$Y_C) > 0) {
    m <- apply(resp$Y_C, 2, max)
    alpha_c <- if (is.null(pp$alpha_c)) m else pp$alpha_c
    out <- out +
      ref_pred_cat(resp$Y_C[i, ], resp$Y_C[idx, , drop = FALSE], alpha_c, m)
  }
  if (ncol(resp$Y_P) > 0) {
    out <- out +
      ref_pred_pois(
        resp$Y_P[i, ],
        resp$Y_P[idx, , drop = FALSE],
        pp$alpha_p,
        pp$beta_p
      )
  }
  out
}

expect_close <- function(object, expected, tol = 1e-10) {
  expect_lt(abs(object - expected), tol)
}

# The slow correctness tests (T6-T8) run only when BMFVS_SLOW_TESTS=true
skip_if_not_slow <- function() {
  skip_on_cran()
  skip_if_not(
    identical(Sys.getenv("BMFVS_SLOW_TESTS"), "true"),
    "slow test: set BMFVS_SLOW_TESTS=true to run"
  )
}

# Log collapsed posterior assembled in R from the cluster marginals
ref_log_post <- function(resp, z, k, alpha) {
  sum(vapply(
    seq_len(k),
    \(h) lgamma(alpha / k + sum(z == h)) + cm_log_m(resp, which(z == h)),
    numeric(1)
  ))
}
