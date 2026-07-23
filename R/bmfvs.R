#' Bayesian Model-Free Variable Screening (BMFVS)
#'
#' @param Y
#' @param X
#' @param n_iter
#' @param n_burnin
#' @param k
#' @param y_types
#' @param alpha
#' @param prior_params
#' @param hyperparams
#' @param kappa
#' @param empirical_bayes
#' @param verbose
#' @param gibbs_fit
#'
#' @returns
#'
#' @export
#' @examples
bmfvs <- function(
  Y,
  X,
  n_iter,
  n_burnin,
  k,
  y_types = NULL,
  alpha = k,
  prior_params = list(),
  hyperparams = list(),
  kappa = c(1 / 2, 1 / 6, 1 / 6, 1 / 6),
  empirical_bayes = TRUE,
  verbose = TRUE,
  gibbs_fit = NULL
) {
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
  if (any(X < 0) || any(X != floor(X))) {
    stop("X must contain non-negative integers 0, ..., d_j - 1")
  }

  # alpha_c gets set in gibbs_mixed_cpp
  prior_params <- utils::modifyList(
    list(
      beta0 = 0.01,
      nu0 = qN + 2,
      eps0 = qN + 2,
      alpha_c = NULL,
      alpha_p = 1,
      beta_p = 0.1
    ),
    prior_params
  )
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
  if (verbose) {
    cat(
      "Stage 1: Gibbs sampler (k =",
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
  if (is.null(gibbs_fit)) {
    fit <- gibbs_mixed_cpp(
      Y_N,
      Y_C,
      Y_P,
      k,
      n_iter,
      n_burnin,
      alpha,
      prior_params$beta0,
      prior_params$nu0,
      prior_params$eps0,
      prior_params$alpha_c,
      prior_params$alpha_p,
      prior_params$beta_p
    )
  } else {
    fit <- gibbs_fit
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
  S <- n_iter - n_burnin
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
    tau_rho = hp$tau_rho
  )
  log_bf1 <- bfs$log_bf1
  log_bf2 <- bfs$log_bf2

  # Stage 2.5: empirical Bayes estimation of kappa --------------------------
  if (empirical_bayes) {
    if (verbose) {
      cat("Empirical Bayes: estimating kappa...\n")
    }
    kappa <- estimate_kappa(log_bf1, log_bf2, kappa_init = kappa)
  }

  # Stage 3: compute posterior probabilities P(H0j) -------------------------
  if (verbose) {
    cat("Stage 3: P(H0j)...\n")
  }
  log_kappa <- log(kappa)
  log_terms <- rbind(
    rep(log_kappa[1], length(log_bf1)),
    log_kappa[2] + as.vector(log_bf1),
    log_kappa[3] + as.vector(log_bf2),
    log_kappa[4] + as.vector(log_bf1 + log_bf2)
  )
  log_denom <- logsumexp_cols(log_terms)
  prob_null_mat <- matrix(exp(log_terms[1, ] - log_denom), nrow = S)
  prob_null <- colMeans(prob_null_mat)
  names(prob_null) <- colnames(X)

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
      fit = fit,
      y_types = resp$y_types,
      y_levels = resp$y_levels
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
