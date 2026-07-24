# Merge user-supplied prior parameters over the weakly-informative defaults
# (Supplemental C.4) and expand per-response entries to the block dimensions
# expected by gibbs_mixed_cpp (alpha_c length qC; alpha_p, beta_p length qP).

# Fit the mixed-kernel Gibbs sampler for a single k
fit_gibbs_mixed <- function(Y_N, Y_C, Y_P, k, n_iter, n_burnin, alpha, pp) {
  gibbs_mixed_cpp(
    Y_N,
    Y_C,
    Y_P,
    k,
    n_iter,
    n_burnin,
    if (is.null(alpha)) -1 else alpha, # -1 -> alpha = k, set in C++
    pp$beta0,
    pp$nu0,
    pp$eps0,
    pp$alpha_c,
    pp$alpha_p,
    pp$beta_p
  )
}

# Core of the WAIC step: fit the baseline mixture for each candidate k,
# compare via WAIC (loo package), return the best k and its fit
select_k_core <- function(
  Y_N,
  Y_C,
  Y_P,
  k_vec,
  n_iter,
  n_burnin,
  alpha,
  pp,
  verbose = TRUE
) {
  list_models <- list()
  list_waic <- list()
  for (k in k_vec) {
    name <- paste0("k=", k)
    if (verbose) {
      cat("  Fitting k =", k, "...\n")
    }
    list_models[[name]] <- fit_gibbs_mixed(
      Y_N,
      Y_C,
      Y_P,
      k,
      n_iter,
      n_burnin,
      alpha,
      pp
    )
    list_waic[[name]] <- suppressWarnings(
      loo::waic(list_models[[name]]$log_lik)
    )
  }

  if (length(k_vec) == 1) {
    best_k <- k_vec
    best_name <- names(list_waic)[1]
    comparison <- list_waic[[1]]
  } else {
    comparison <- loo::loo_compare(list_waic)
    best_name <- if (is.data.frame(comparison) && !is.null(comparison$model)) {
      comparison$model[1]
    } else {
      rownames(comparison)[1]
    }
    best_k <- k_vec[match(best_name, names(list_waic))]
  }
  if (verbose) {
    print(comparison)
    cat("Selected k =", best_k, "\n")
  }

  list(k = best_k, fit = list_models[[best_name]], comparison = comparison)
}

#' Select the number of mixture components k via WAIC
#'
#' Fits the baseline mixture model for each value in 'k' and compares the fits
#' using WAIC. The returned fit can be inspected with [plot_gibbs()] and
#' passed to [bmfvs()] to skip the Gibbs sampling stage.
#'
#' @param Y
#' @param k
#' @param n_iter
#' @param n_burnin
#' @param y_types
#' @param alpha
#' @param prior_params
#' @param verbose
#'
#' @returns
#'
#' @export
#' @examples
#' NULL
select_k <- function(
  Y,
  k,
  n_iter,
  n_burnin,
  y_types = NULL,
  alpha = NULL,
  prior_params = list(),
  verbose = TRUE
) {
  resp <- split_response(Y, y_types)
  pp <- expand_prior_params(
    prior_params,
    ncol(resp$Y_N),
    ncol(resp$Y_C),
    ncol(resp$Y_P)
  )
  select_k_core(
    resp$Y_N,
    resp$Y_C,
    resp$Y_P,
    k,
    n_iter,
    n_burnin,
    alpha,
    pp,
    verbose
  )
}
