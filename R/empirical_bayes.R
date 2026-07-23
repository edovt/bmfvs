# Log-sum-exp helper (vectorized over columns, matrix version)
# x is a 4 x n matrix of log values, returns length-n vector
logsumexp_cols <- function(x) {
  m <- apply(x, 2, max)
  m + log(colSums(exp(x - rep(m, each = nrow(x)))))
}

# Empirical Bayes estimation of the prior probabilities kappa
estimate_kappa <- function(
  log_bf1,
  log_bf2,
  kappa_init = c(1 / 2, 1 / 6, 1 / 6, 1 / 6),
  max_iter = 100,
  tol = 1e-4
) {
  S <- nrow(log_bf1)
  p <- ncol(log_bf1)

  kappa <- kappa_init

  for (iter in seq_len(max_iter)) {
    log_kappa <- log(kappa)

    # 4 x (S*p) matrix of log-terms
    log_terms <- rbind(
      rep(log_kappa[1], S * p),
      log_kappa[2] + as.vector(log_bf1),
      log_kappa[3] + as.vector(log_bf2),
      log_kappa[4] + as.vector(log_bf1 + log_bf2)
    )
    log_denom <- logsumexp_cols(log_terms)

    kappa_new <- c(
      sum(exp(log_terms[1, ] - log_denom)),
      sum(exp(log_terms[2, ] - log_denom)),
      sum(exp(log_terms[3, ] - log_denom)),
      sum(exp(log_terms[4, ] - log_denom))
    ) /
      (p * S)

    if (max(abs(kappa_new - kappa)) < tol) {
      return(kappa_new)
    }
    kappa <- kappa_new
  }

  kappa
}
