#' Bayesian Model-Free Variable Screening (BMFVS)
#'
#' @param Y
#' @param X
#' @param n_iter
#' @param n_burnin
#' @param k Number of mixture components. A single value fits that k; a
#'   vector selects among its values by WAIC; `NULL` (the default) runs the
#'   adaptive WAIC search of [select_k()] (start at `1:k_init`, extend by
#'   `k_step` while the best k is in the top window, up to `k_max`). Ignored
#'   when `gibbs_fit` is supplied.
#' @param y_types
#' @param alpha
#' @param prior_params
#' @param hyperparams
#' @param kappa
#' @param empirical_bayes
#' @param verbose
#' @param gibbs_fit
#' @param sampler Stage-1 sampler for the mixture (see [select_k()]):
#'   `"nonreversible"` (default), `"marginal"` or `"conditional"`. Ignored
#'   when `gibbs_fit` is supplied.
#' @param k_init,k_step,k_max Settings of the adaptive search used when
#'   `k = NULL`; see [select_k()].
#'
#' @returns An object of class `"bmfvs"`, a list with (among others)
#' \describe{
#'   \item{`prob_null`}{Posterior probabilities \eqn{P(H_{0j} \mid Y, X_j)}.
#'     The screening rule of the paper keeps the predictors with `prob_null`
#'     below a threshold.}
#'   \item{`prob_h1`}{\eqn{p \times 3} matrix of the posterior probabilities
#'     of the three alternatives (weights, kernel parameters, both).}
#'   \item{`kappa`}{Prior probabilities of \eqn{(H_0, H_{11}, H_{12},
#'     H_{13})}, estimated by empirical Bayes unless `empirical_bayes = FALSE`.}
#'   \item{`log_prob_null`}{\eqn{\log P(H_{0j} \mid Y, X_j)} computed by
#'     log-sum-exp; finite where `prob_null` underflows to 0.}
#'   \item{`log_odds_h1`}{Posterior log odds
#'     \eqn{\log P(H_{1j} \mid Y, X_j) - \log P(H_{0j} \mid Y, X_j)}, i.e.
#'     `prob_null` on the logit scale, computed so that it stays finite and
#'     ordered where `prob_null` rounds to exactly 0 or 1.}
#'   \item{`rank`}{Rank of each predictor by decreasing `log_odds_h1`
#'     (1 = strongest evidence; ties share the smallest rank). It equals the
#'     ranking by increasing `prob_null` wherever `prob_null` separates the
#'     predictors; use it to retain a fixed number of predictors.}
#'   \item{`eb_collapsed`}{`TRUE` (with a warning) when empirical Bayes
#'     converges to the \eqn{\kappa = (1, 0, 0, 0)} corner, i.e. when the
#'     Bayes factors averaged over all predictors and posterior samples
#'     (`eb_log_mean_bf`) are all below 1. Posterior null probabilities are
#'     then close to 1, thresholding `prob_null` selects few or no
#'     predictors, and `rank` orders predictors by their evidence only.}
#'   \item{`eb_log_mean_bf`}{log of the three Bayes factors averaged over
#'     predictors and posterior samples.}
#'   \item{`k`, `k_candidates`, `waic`, `fit`, `log_bf1`, `log_bf2`,
#'     `hyperparams`}{Stage-1 and stage-2 details.}
#' }
#'
#' @export
#' @examples
#' NULL
bmfvs <- function(
  Y,
  X,
  n_iter,
  n_burnin,
  k = NULL,
  y_types = NULL,
  alpha = NULL,
  prior_params = list(),
  hyperparams = list(),
  kappa = c(1 / 2, 1 / 6, 1 / 6, 1 / 6),
  empirical_bayes = TRUE,
  verbose = TRUE,
  gibbs_fit = NULL,
  backend = c("cpp", "r"),
  n_threads = default_threads(),
  sampler = c("nonreversible", "marginal", "conditional"),
  k_init = 20L,
  k_step = 10L,
  k_max = NULL
) {
  backend <- match.arg(backend)
  sampler <- match.arg(sampler)
  # Stage 0: checks and response splitting ----------------------------------
  resp <- split_response(Y, y_types)
  Y_N <- resp$Y_N
  Y_C <- resp$Y_C
  Y_P <- resp$Y_P
  qN <- ncol(Y_N)
  qC <- ncol(Y_C)
  qP <- ncol(Y_P)
  n <- nrow(Y_N)

  X <- as.matrix(X)
  if (nrow(X) != n) {
    stop("X must have the same number of rows as Y")
  }

  # TODO: check correctness of X
  if (any(X < 0) || any(X != floor(X))) {
    stop("X must contain non-negative integers 0, ..., d_j - 1")
  }

  # Defaults params merged and expanded
  prior_params <- expand_prior_params(prior_params, qN, qC, qP)
  hp <- utils::modifyList(
    list(
      tau_omega = NULL,
      tau_mu = NULL,
      tau_Sigma = NULL,
      tau_psi = NULL,
      tau_rho = NULL
    ),
    hyperparams
  )

  # Stage 1: Gibbs sampler of marginal posterior ----------------------------
  # If k is a vector (or NULL, adaptive search), the WAIC step selects k
  waic <- NULL
  k_candidates <- NULL
  if (!is.null(gibbs_fit)) {
    fit <- if (!is.null(gibbs_fit$fit)) gibbs_fit$fit else gibbs_fit
    k <- ncol(fit$w)
    if (verbose) {
      cat(
        "Stage 1: using supplied fit (k =",
        k,
        if (!is.null(attr(fit, "sampler"))) {
          paste0(", sampler = ", attr(fit, "sampler"))
        },
        ")...\n"
      )
    }
  } else if (is.null(k) || length(k) > 1) {
    if (verbose) {
      if (is.null(k)) {
        cat(
          "Stage 1: selecting k via adaptive WAIC search (start 1 ...",
          k_init,
          ", step",
          k_step,
          ", max",
          if (is.null(k_max)) paste0("min(n, 100) = ", min(n, 100)) else k_max,
          ", sampler =",
          sampler,
          ")...\n"
        )
      } else {
        cat(
          "Stage 1: selecting k via WAIC (candidates:",
          k,
          ", sampler =",
          sampler,
          ")...\n"
        )
      }
    }
    sel <- select_k_core(
      Y_N,
      Y_C,
      Y_P,
      k,
      n_iter,
      n_burnin,
      alpha,
      prior_params,
      sampler,
      verbose,
      k_init = k_init,
      k_step = k_step,
      k_max = k_max
    )
    k <- sel$k
    fit <- sel$fit
    waic <- sel$comparison
    k_candidates <- sel$k_candidates
  } else {
    if (verbose) {
      cat(
        "Stage 1:",
        sampler,
        "sampler (k =",
        k,
        ", qN =",
        qN,
        ", qC =",
        qC,
        ", qP =",
        qP,
        ", n_iter =",
        n_iter,
        ")...\n"
      )
    }
    fit <- fit_gibbs_mixed(
      Y_N,
      Y_C,
      Y_P,
      k,
      n_iter,
      n_burnin,
      alpha,
      prior_params,
      sampler
    )
  }

  # Stage 1.5: hyperparameter selection (per block, common fit) -------------
  if (is.null(hp$tau_omega)) {
    if (verbose) {
      cat("Hyperparameter selection: estimating tau_omega...\n")
    }
    hp$tau_omega <- select_tau_omega(fit$w)
  }
  if (qN > 0 && (is.null(hp$tau_mu) || is.null(hp$tau_Sigma))) {
    if (verbose) {
      cat("Hyperparameter selection: estimating (tau_mu, tau_Sigma)...\n")
    }
    tau_theta <- select_tau_gaussian(fit$mu, fit$Sigma)
    if (is.null(hp$tau_mu)) {
      hp$tau_mu <- tau_theta[1]
    }
    if (is.null(hp$tau_Sigma)) {
      hp$tau_Sigma <- tau_theta[2]
    }
  }
  if (qC > 0 && is.null(hp$tau_psi)) {
    if (verbose) {
      cat("Hyperparameter selection: estimating tau_psi...\n")
    }
    hp$tau_psi <- select_tau_multinomial(fit$psi)
  }
  if (qP > 0 && is.null(hp$tau_rho)) {
    if (verbose) {
      cat("Hyperparameter selection: estimating tau_rho...\n")
    }
    hp$tau_rho <- select_tau_poisson(fit$rho)
  }

  # Stage 2: compute log Bayes factors --------------------------------------
  S <- nrow(fit$z)
  p <- ncol(X)
  if (verbose) {
    cat("Stage 2: Bayes factors (p =", p, ", S =", S, ")...\n")
  }
  bfs <- compute_log_bfs(
    fit,
    Y_N = if (qN > 0) Y_N else NULL,
    Y_C = if (qC > 0) Y_C else NULL,
    Y_P = if (qP > 0) Y_P else NULL,
    X = X,
    tau_omega = hp$tau_omega,
    tau_mu = hp$tau_mu,
    tau_Sigma = hp$tau_Sigma,
    tau_psi = hp$tau_psi,
    tau_rho = hp$tau_rho,
    backend = backend,
    n_threads = n_threads
  )
  log_bf1 <- bfs$log_bf1
  log_bf2 <- bfs$log_bf2

  # Stage 2.5: empirical Bayes estimation of kappa --------------------------
  if (empirical_bayes) {
    if (verbose) {
      cat("Empirical Bayes: estimating kappa...\n")
    }
    kappa <- estimate_kappa(
      log_bf1,
      log_bf2,
      kappa_init = kappa,
      backend = backend
    )
  }

  # Stage 3: compute posterior probabilities P(H0j) -------------------------
  if (verbose) {
    cat("Stage 3: P(H0j)...\n")
  }
  post <- posterior_probs(log_bf1, log_bf2, kappa, backend = backend)
  prob_null <- post$prob_null
  names(prob_null) <- colnames(X)
  # log P(H0j) by log-sum-exp: prob_null underflows to exactly 0 for strong
  # signals, which produces ties in the ranking; log_prob_null does not
  log_prob_null <- post$log_prob_null
  names(log_prob_null) <- colnames(X)
  prob_h1 <- post$prob_h1
  rownames(prob_h1) <- colnames(X)
  # Posterior log odds of H1j, logit(1 - prob_null) computed in log space: the
  # same ordering as prob_null wherever prob_null separates predictors, and
  # still ordered where prob_null rounds to exactly 0 or 1. Predictors are
  # ranked by it (1 = strongest evidence for dependence).
  log_odds_h1 <- post$log_odds_h1
  names(log_odds_h1) <- colnames(X)
  rank <- base::rank(-log_odds_h1, ties.method = "min")

  eb_log_mean_bf <- log_mean_bf(log_bf1, log_bf2, backend = backend)
  eb_collapsed <- empirical_bayes && eb_corner(eb_log_mean_bf)
  if (eb_collapsed) {
    warning(
      "Empirical Bayes is converging to kappa_0 = 1 (kappa = ",
      paste(signif(kappa, 3), collapse = ", "),
      "): the Bayes factors averaged over all predictors and posterior ",
      "samples are below 1 (log means ",
      paste(round(eb_log_mean_bf, 2), collapse = ", "),
      "), so the data give no overall evidence that any predictor is ",
      "associated with the response. Posterior null probabilities are close ",
      "to 1 and thresholding prob_null selects few or no predictors. `rank` ",
      "(from `log_odds_h1`) still orders the predictors by their evidence if ",
      "a fixed-size screened set is required, but no predictor is ",
      "individually supported by the data.",
      call. = FALSE
    )
  }

  if (verbose) {
    cat("Done.\n")
  }

  structure(
    list(
      prob_null = prob_null,
      kappa = kappa,
      hyperparams = hp,
      log_bf1 = log_bf1,
      log_bf2 = log_bf2,
      k = k,
      waic = waic,
      fit = fit,
      y_types = resp$y_types,
      y_levels = resp$y_levels,
      # Additions (appended so that existing elements keep their positions)
      log_prob_null = log_prob_null,
      prob_h1 = prob_h1,
      # Values of k fitted by the WAIC step (NULL when k was fixed or a fit
      # was supplied)
      k_candidates = k_candidates,
      log_odds_h1 = log_odds_h1,
      rank = rank,
      eb_collapsed = eb_collapsed,
      eb_log_mean_bf = eb_log_mean_bf
    ),
    class = "bmfvs"
  )
}

# Stage 0 helper: split response into three blocks
# Column types are inferred from the column class unless overridden
split_response <- function(Y, y_types = NULL) {
  if (is.vector(Y) || is.factor(Y)) {
    Y <- data.frame(Y = Y)
  }
  if (is.matrix(Y)) {
    Y <- as.data.frame(Y)
  }
  if (!is.data.frame(Y)) {
    stop("Y must be a data.frame, matrix, or vector")
  }
  n <- nrow(Y)
  q <- ncol(Y)
  if (anyNA(Y)) {
    stop("Y contains missing values.")
  }

  # Determine the type of each column
  if (is.null(y_types)) {
    y_types <- vapply(
      Y,
      function(col) {
        if (is.factor(col) || is.character(col) || is.logical(col)) {
          "categorical"
        } else if (is.integer(col)) {
          "poisson"
        } else if (is.numeric(col)) {
          "gaussian"
        } else {
          stop("Unsupported column class in Y: ", class(col)[1])
        }
      },
      character(1)
    )
  } else {
    if (length(y_types) == 1) {
      y_types <- rep(y_types, q)
    }
    if (length(y_types) != q) {
      stop("y_types must have length 1 or ncol(Y)")
    }
    y_types <- vapply(
      y_types,
      \(x) match.arg(x, c("gaussian", "categorical", "poisson")),
      character(1)
    )
  }
  names(y_types) <- names(Y)

  # Build the three blocks plus verification
  idx_N <- which(y_types == "gaussian")
  idx_C <- which(y_types == "categorical")
  idx_P <- which(y_types == "poisson")

  ## Continuous columns
  Y_N <- matrix(0, n, length(idx_N), dimnames = list(NULL, names(Y)[idx_N]))
  for (j in seq_along(idx_N)) {
    col <- Y[[idx_N[j]]]
    if (!is.numeric(col)) {
      stop("Column '", names(Y)[idx_N[j]], "' is not numeric (gaussian)")
    }
    Y_N[, j] <- as.numeric(col)
  }

  # Categorical columns are recoded to 1, ..., m_t (unused levels dropped)
  Y_C <- matrix(0L, n, length(idx_C), dimnames = list(NULL, names(Y)[idx_C]))
  y_levels <- list()
  for (j in seq_along(idx_C)) {
    f <- droplevels(as.factor(Y[[idx_C[j]]]))
    if (nlevels(f) < 2) {
      stop("Column '", names(Y)[idx_C[j]], "' has fewer than 2 levels.")
    }
    Y_C[, j] <- as.integer(f)
    y_levels[[names(Y)[idx_C[j]]]] <- levels(f)
  }

  ## Count-valued columns
  Y_P <- matrix(0, n, length(idx_P), dimnames = list(NULL, names(Y)[idx_P]))
  for (j in seq_along(idx_P)) {
    col <- as.numeric(Y[[idx_P[j]]])
    if (any(col < 0) || any(col != floor(col))) {
      stop(
        "Column '",
        names(Y)[idx_P[j]],
        "' must contain non-negative integers (poisson)"
      )
    }
    Y_P[, j] <- col
  }

  list(Y_N = Y_N, Y_C = Y_C, Y_P = Y_P, y_types = y_types, y_levels = y_levels)
}

expand_prior_params <- function(prior_params, qN, qC, qP) {
  pp <- utils::modifyList(
    list(
      beta0 = 0.01,
      nu0 = qN + 2,
      eps0 = qN + 2,
      alpha_c = NULL, # set in gibbs_mixed_cpp
      alpha_p = 1,
      beta_p = 0.1
    ),
    prior_params
  )
  if (!is.null(pp$alpha_c) && qC > 0) {
    pp$alpha_c <- as.numeric(rep(pp$alpha_c, length.out = qC))
  }
  if (qP > 0) {
    pp$alpha_p <- rep(pp$alpha_p, length.out = qP)
    pp$beta_p <- rep(pp$beta_p, length.out = qP)
  }
  pp
}
