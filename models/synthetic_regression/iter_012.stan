// Iter 12: Piecewise linear (knot at x=0) + predictor-dependent heteroscedasticity
// sigma_n = sigma_j * exp(xi_j * x^2) -- wider at extreme predictor values
data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> unit_train;
  array[N_test] int<lower=1,upper=J> unit_test;
  vector[N_train] predictor_train;
  vector[N_test] predictor_test;
  vector[N_train] response_train;
  vector[N_test] response_test;
}
transformed data {
  vector[N_train] x_neg_train = fmin(predictor_train, rep_vector(0.0, N_train));
  vector[N_train] x_pos_train = fmax(predictor_train, rep_vector(0.0, N_train));
  vector[N_test]  x_neg_test  = fmin(predictor_test,  rep_vector(0.0, N_test));
  vector[N_test]  x_pos_test  = fmax(predictor_test,  rep_vector(0.0, N_test));
}
parameters {
  vector[3] mu;                    // [mu_alpha, mu_beta_neg, mu_beta_pos]
  vector<lower=0>[3] tau;
  cholesky_factor_corr[3] L;
  matrix[3, J] z;
  // Hierarchical residual sigma (base level)
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[J] z_sigma;
  // Predictor-dependent heteroscedasticity: log(sigma_n) += xi_j * x^2
  real mu_xi;
  real<lower=0> sigma_xi;
  vector[J] z_xi;
}
transformed parameters {
  matrix[J, 3] abg;
  {
    matrix[3, 3] L_Sigma = diag_pre_multiply(tau, L);
    abg = (L_Sigma * z)';
  }
  vector[J] alpha    = mu[1] + abg[, 1];
  vector[J] beta_neg = mu[2] + abg[, 2];
  vector[J] beta_pos = mu[3] + abg[, 3];
  vector<lower=0>[J] sigma_base = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
  vector[J] xi = mu_xi + sigma_xi * z_xi;  // can be negative (less var at extremes)
}
model {
  mu ~ normal(0, 5);
  tau ~ exponential(1);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ exponential(1);
  z_sigma ~ std_normal();
  mu_xi ~ normal(0, 0.5);         // prior: no predictor-dependent variance on average
  sigma_xi ~ exponential(2);      // tight: small unit-to-unit variation in xi
  z_xi ~ std_normal();
  for (n in 1:N_train) {
    int j = unit_train[n];
    real mu_n = alpha[j] + beta_neg[j] * x_neg_train[n] + beta_pos[j] * x_pos_train[n];
    real sigma_n = sigma_base[j] * exp(xi[j] * predictor_train[n]^2);
    response_train[n] ~ normal(mu_n, sigma_n);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    int j = unit_test[n];
    real mu_n = alpha[j] + beta_neg[j] * x_neg_test[n] + beta_pos[j] * x_pos_test[n];
    real sigma_n = sigma_base[j] * exp(xi[j] * predictor_test[n]^2);
    log_lik[n] = normal_lpdf(response_test[n] | mu_n, sigma_n);
  }
}
