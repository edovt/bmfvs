#' Plot the marginal fitted baseline mixture against the observed response
#'
#' Draws, for each response column, the observed marginal distribution with the
#' posterior mean mixture marginal overlaid.
#'
#' @param Y
#' @param fit either a raw Gibbs fit, the output of [select_k()],
#'   or the output of [bmfvs()]
#' @param y_types
#' @param breaks
#'
#' @returns
#'
#' @export
#' @examples
#' NULL
plot_gibbs <- function(Y, fit, y_types = NULL, breaks = 30) {
  # Accept bmfvs() / select_k() outputs as well as a raw Gibbs fit
  if (!is.null(fit$fit)) {
    fit <- fit$fit
  }

  resp <- split_response(Y, y_types)
  y_types <- resp$y_types
  q <- length(y_types)
  n <- nrow(resp$Y_N)
  S <- nrow(fit$w)
  k <- ncol(fit$w)

  if (q > 1) {
    nc <- ceiling(sqrt(q))
    nr <- ceiling(q / nc)
    old_par <- graphics::par(mfrow = c(nr, nc))
    on.exit(graphics::par(old_par))
  }

  iN <- 0L
  iC <- 0L
  iP <- 0L
  for (t in seq_len(q)) {
    name <- names(y_types)[t]

    if (y_types[t] == "gaussian") {
      iN <- iN + 1L
      y <- resp$Y_N[, iN]
      y_grid <- seq(min(y) - 1, max(y) + 1, length.out = 400)

      # Average the mixture marginal pdf over posterior samples
      pdf_est <- numeric(length(y_grid))
      for (s in seq_len(S)) {
        for (h in seq_len(k)) {
          pdf_est <- pdf_est +
            fit$w[s, h] *
              stats::dnorm(
                y_grid,
                fit$mu[s, h, iN],
                sqrt(fit$Sigma[s, h, iN, iN])
              )
        }
      }
      pdf_est <- pdf_est / S

      hh <- graphics::hist(y, breaks = breaks, plot = FALSE)
      graphics::hist(
        y,
        breaks = breaks,
        probability = TRUE,
        col = "grey85",
        border = "white",
        main = paste0(name, " (gaussian)"),
        xlab = "y",
        ylim = c(0, max(pdf_est, hh$density) * 1.1)
      )
      graphics::lines(y_grid, pdf_est, col = "blue", lwd = 2)
    } else if (y_types[t] == "categorical") {
      iC <- iC + 1L
      y <- resp$Y_C[, iC]
      m_t <- dim(fit$psi[[iC]])[3]

      # Average the mixture pmf over posterior samples
      pmf_est <- vapply(
        seq_len(m_t),
        \(l) mean(rowSums(fit$w * fit$psi[[iC]][,, l])),
        numeric(1)
      )

      emp <- tabulate(y, nbins = m_t) / n
      bar_x <- graphics::barplot(
        emp,
        names.arg = resp$y_levels[[name]],
        col = "grey85",
        border = "black",
        main = paste0(name, " (categorical)"),
        xlab = "y",
        ylab = "Probability",
        ylim = c(0, max(pmf_est, emp) * 1.1)
      )
      graphics::lines(
        bar_x,
        pmf_est,
        col = "blue",
        lwd = 2,
        type = "b",
        pch = 16
      )
    } else {
      iP <- iP + 1L
      y <- resp$Y_P[, iP]
      y_grid <- 0:(max(y) + 2)

      # Average the mixture pmf over posterior samples
      pmf_est <- numeric(length(y_grid))
      for (s in seq_len(S)) {
        for (h in seq_len(k)) {
          pmf_est <- pmf_est +
            fit$w[s, h] * stats::dpois(y_grid, lambda = fit$rho[s, h, iP])
        }
      }
      pmf_est <- pmf_est / S

      emp <- tabulate(y + 1L, nbins = length(y_grid)) / n
      bar_x <- graphics::barplot(
        emp,
        names.arg = y_grid,
        col = "grey85",
        border = "black",
        main = paste0(name, " (poisson)"),
        xlab = "y",
        ylab = "Probability",
        ylim = c(0, max(pmf_est, emp) * 1.1)
      )
      graphics::lines(
        bar_x,
        pmf_est,
        col = "blue",
        lwd = 2,
        type = "b",
        pch = 16
      )
    }
  }

  invisible(NULL)
}
