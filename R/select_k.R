# Merge user-supplied prior parameters over the weakly-informative defaults
# (Supplemental C.4) and expand per-response entries to the block dimensions
# expected by gibbs_mixed_cpp (alpha_c length qC; alpha_p, beta_p length qP).

# Fit the mixed-kernel mixture for a single k. The collapsed samplers
# ("nonreversible", "marginal") update the allocations with (w, theta)
# integrated out and draw (w, theta) | z at saved iterations; "conditional"
# alternates z | (w, theta) and (w, theta) | z. All return the same output,
# with the sampler name in attr(fit, "sampler").
fit_gibbs_mixed <- function(
  Y_N,
  Y_C,
  Y_P,
  k,
  n_iter,
  n_burnin,
  alpha,
  pp,
  sampler
) {
  alpha <- if (is.null(alpha)) -1 else alpha # -1 -> alpha = k, set in C++
  fit <- if (sampler == "conditional") {
    gibbs_mixed_cpp(
      Y_N,
      Y_C,
      Y_P,
      k,
      n_iter,
      n_burnin,
      alpha,
      pp$beta0,
      pp$nu0,
      pp$eps0,
      pp$alpha_c,
      pp$alpha_p,
      pp$beta_p
    )
  } else {
    collapsed_mixed_cpp(Y_N, Y_C, Y_P, k, n_iter, n_burnin, alpha, pp, sampler)
  }
  attr(fit, "sampler") <- sampler
  fit
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
  sampler,
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
      pp,
      sampler
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
#' @param sampler Stage-1 sampler for the mixture. `"nonreversible"` (the
#'   default) is the lifted non-reversible sampler of Ascolani and Zanella
#'   (2025) on the collapsed posterior of the allocations, `"marginal"` is the
#'   collapsed (marginal) Gibbs sampler, and `"conditional"` is the Gibbs
#'   sampler that alternates between the allocations and the mixture
#'   parameters. The collapsed samplers draw the weights and kernel parameters
#'   given the allocations at every saved iteration. For all three, one
#'   iteration is n single-observation updates of the allocations.
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
  verbose = TRUE,
  sampler = c("nonreversible", "marginal", "conditional")
) {
  sampler <- match.arg(sampler)
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
    sampler,
    verbose
  )
}
