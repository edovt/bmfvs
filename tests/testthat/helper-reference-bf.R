# Plain-R transcription of Proposition 2.1 and eqs. (gauss_bf2),
# (prod_mulnom_bf2), (prod_pois_bf2), cell by cell, for the BF tests

ref_lbeta <- function(a) sum(lgamma(a)) - lgamma(sum(a))
ref_lmvg <- function(a, q) q * (q - 1) / 4 * log(pi) + sum(lgamma(a + (1 - seq_len(q)) / 2))
ref_ldet <- function(A) as.numeric(determinant(A)$modulus)

ref_bf1 <- function(z, w, xj, tau, k) {
  out <- -sum(tabulate(z, k) * log(w))
  for (l in 0:max(xj)) {
    out <- out + ref_lbeta(tabulate(z[xj == l], k) + tau * w) - ref_lbeta(tau * w)
  }
  out
}

ref_bf2_gauss <- function(z, Y, xj, mu, Sigma, tmu, tS, k) {
  q <- ncol(Y); out <- 0
  for (l in 0:max(xj)) for (h in seq_len(k)) {
    idx <- which(xj == l & z == h); nn <- length(idx)
    Lam <- tS * Sigma[[h]]
    if (nn > 0) {
      yb <- colMeans(Y[idx, , drop = FALSE])
      Lam <- Lam + tmu * nn / (tmu + nn) * tcrossprod(yb - mu[[h]]) +
        crossprod(sweep(Y[idx, , drop = FALSE], 2, yb))
    }
    out <- out + nn * q / 2 * log(2) + ref_lmvg((tS + nn) / 2, q) +
      tS / 2 * ref_ldet(tS * Sigma[[h]]) + q / 2 * log(tmu) - ref_lmvg(tS / 2, q) -
      (tS + nn) / 2 * ref_ldet(Lam) - q / 2 * log(tmu + nn)
  }
  for (h in seq_len(k)) {
    idx <- which(z == h); if (!length(idx)) next
    D <- sweep(Y[idx, , drop = FALSE], 2, mu[[h]])
    out <- out + length(idx) / 2 * ref_ldet(Sigma[[h]]) + 0.5 * sum((D %*% solve(Sigma[[h]])) * D)
  }
  out
}

ref_bf2_cat <- function(z, Yc, xj, psi, tau, k) {
  out <- 0
  for (t in seq_along(psi)) for (h in seq_len(k)) {
    P <- psi[[t]][h, ]; m <- length(P)
    out <- out - sum(tabulate(Yc[z == h, t], m) * log(P))
    for (l in 0:max(xj)) {
      out <- out + ref_lbeta(tabulate(Yc[z == h & xj == l, t], m) + tau * P) - ref_lbeta(tau * P)
    }
  }
  out
}

ref_bf2_pois <- function(z, Yp, xj, rho, tau, k) {
  out <- 0
  for (h in seq_len(k)) for (t in seq_len(ncol(Yp))) {
    r <- rho[h, t]; a <- tau * r
    out <- out + sum(z == h) * r - sum(Yp[z == h, t]) * log(r)
    for (l in 0:max(xj)) {
      idx <- z == h & xj == l; S <- sum(Yp[idx, t])
      out <- out + lgamma(a + S) + a * log(tau) - lgamma(a) - (a + S) * log(tau + sum(idx))
    }
  }
  out
}
