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

# Validate the settings of the adaptive search used when k = NULL and resolve
# the default cap k_max = min(n, 100).
check_k_search <- function(n, k_init, k_step, k_max) {
  is_count <- function(x) {
    is.numeric(x) && length(x) == 1 && !is.na(x) && x >= 1 && x == round(x)
  }
  if (!is_count(k_init)) stop("k_init must be a single integer >= 1")
  if (!is_count(k_step)) stop("k_step must be a single integer >= 1")
  if (is.null(k_max)) {
    k_max <- min(n, 100L)
  }
  if (!is_count(k_max)) stop("k_max must be a single integer >= 1")
  list(
    k_init = as.integer(min(k_init, k_max)),
    k_step = as.integer(k_step),
    k_max = as.integer(k_max)
  )
}

# Core of the WAIC step: fit the baseline mixture for each candidate k,
# compare via WAIC (loo package), return the best k and its fit.
#
# k_vec = NULL runs the adaptive search: fit k = 1, ..., k_init; while the
# WAIC-best k lies in the top window (k_best >= upper - k_step) and upper <
# k_max, fit the next k_step values (upper + 1, ..., upper + k_step) and
# re-select over everything fitted so far. Each k is fitted once, in
# increasing order, so the first window reproduces k_vec = 1:k_init exactly
# (same fits, same RNG stream), and when no extension is needed the result
# is identical to k_vec = 1:k_init.
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
  verbose = TRUE,
  k_init = 20L,
  k_step = 10L,
  k_max = NULL
) {
  # Only the fit with the largest elpd_waic so far is kept, so at most two
  # fits are alive at a time instead of length(k_vec). The rule (first
  # maximum, NA elpd never preferred) is the order used by loo_compare()
  # (stable decreasing order, NA last), so the retained fit is the one that
  # loo_compare() ranks first; this is checked below.
  list_waic <- list()
  best_fit <- NULL
  best_fit_name <- NULL
  best_k_so_far <- NA_integer_
  best_elpd <- NA_real_
  scan_k <- function(ks) {
    for (k in ks) {
      name <- paste0("k=", k)
      if (verbose) {
        cat("  Fitting k =", k, "...\n")
      }
      fit_k <- fit_gibbs_mixed(
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
      list_waic[[name]] <<- suppressWarnings(
        loo::waic(fit_k$log_lik)
      )
      elpd_k <- list_waic[[name]]$estimates["elpd_waic", "Estimate"]
      if (
        is.null(best_fit) ||
          (!is.na(elpd_k) && (is.na(best_elpd) || elpd_k > best_elpd))
      ) {
        best_fit <<- fit_k
        best_fit_name <<- name
        best_k_so_far <<- k
        best_elpd <<- elpd_k
      }
      rm(fit_k)
    }
  }

  if (is.null(k_vec)) {
    ks <- check_k_search(nrow(Y_N), k_init, k_step, k_max)
    upper <- ks$k_init
    scan_k(seq_len(upper))
    while (best_k_so_far >= upper - ks$k_step && upper < ks$k_max) {
      new_upper <- min(upper + ks$k_step, ks$k_max)
      if (verbose) {
        cat(
          "  Best k so far is", best_k_so_far,
          "(within k_step =", ks$k_step, "of the upper end", upper,
          "): extending candidates to k =", upper + 1, "...", new_upper, "\n"
        )
      }
      scan_k((upper + 1):new_upper)
      upper <- new_upper
    }
    if (best_k_so_far >= upper - ks$k_step && upper >= ks$k_max) {
      warning(
        "select_k: the WAIC-best k (", best_k_so_far, ") is within k_step of ",
        "k_max = ", ks$k_max, "; consider a larger k_max.",
        call. = FALSE
      )
    }
    k_vec <- seq_len(upper)
  } else {
    scan_k(k_vec)
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

  if (!identical(best_name, best_fit_name)) {
    stop(
      "select_k: retained fit (", best_fit_name, ") differs from the model ",
      "ranked first by loo_compare() (", best_name, ")"
    )
  }
  list(
    k = best_k,
    fit = best_fit,
    comparison = comparison,
    k_candidates = k_vec
  )
}

#' Select the number of mixture components k via WAIC
#'
#' Fits the baseline mixture model for each candidate number of components and
#' compares the fits using WAIC. The returned fit can be inspected with
#' [plot_gibbs()] and passed to [bmfvs()] to skip the Gibbs sampling stage.
#'
#' @param Y
#' @param k Candidate numbers of components. A numeric vector fits exactly
#'   those values (a single value skips the comparison). `NULL` (the default)
#'   runs an adaptive search: k = 1, ..., `k_init` are fitted first; while the
#'   WAIC-best k is within `k_step` of the largest value fitted so far, the
#'   next `k_step` values are fitted and the selection is repeated over all
#'   fits, up to `k_max`. Each k is fitted once, in increasing order, so when
#'   no extension is needed the result is identical to `k = 1:k_init`.
#' @param k_init,k_step,k_max Settings of the adaptive search used when
#'   `k = NULL`: the initial candidates `1:k_init` (default 20), the size of
#'   each extension and of the top window that triggers it (default 10), and
#'   the largest k that may be fitted (default `min(n, 100)`). A warning is
#'   given if the search stops at `k_max` with the best k still in the top
#'   window. Ignored when `k` is supplied.
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
#' @returns A list with the selected `k`, its `fit`, the WAIC `comparison`
#'   ([loo::loo_compare()] output, or a single `waic` object) and
#'   `k_candidates`, the values of k that were fitted.
#'
#' @export
#' @examples
#' NULL
select_k <- function(
  Y,
  k = NULL,
  n_iter,
  n_burnin,
  y_types = NULL,
  alpha = NULL,
  prior_params = list(),
  verbose = TRUE,
  sampler = c("nonreversible", "marginal", "conditional"),
  k_init = 20L,
  k_step = 10L,
  k_max = NULL
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
    verbose,
    k_init = k_init,
    k_step = k_step,
    k_max = k_max
  )
}
