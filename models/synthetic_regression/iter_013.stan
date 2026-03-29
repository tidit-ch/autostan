// Iter 13: Piecewise linear with LEARNED shared knot location k + unit-specific sigma
// k ~ normal(0, 1) is estimated from data; may improve over fixed knot at 0
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
parameters {
  vector[3] mu;                    // [mu_alpha, mu_beta_neg, mu_beta_pos]
  vector<lower=0>[3] tau;
  cholesky_factor_corr[3] L;
  matrix[3, J] z;
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[J] z_sigma;
  real k;                          // shared knot location (learned from data)
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
  vector<lower=0>[J] sigma = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}
model {
  mu ~ normal(0, 5);
  tau ~ exponential(1);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ exponential(1);
  z_sigma ~ std_normal();
  k ~ normal(0, 1);               // knot prior: likely near 0 but can shift
  for (n in 1:N_train) {
    int j = unit_train[n];
    real x = predictor_train[n];
    real x_neg = fmin(x - k, 0.0);
    real x_pos = fmax(x - k, 0.0);
    real mu_n = alpha[j] + beta_neg[j] * x_neg + beta_pos[j] * x_pos;
    response_train[n] ~ normal(mu_n, sigma[j]);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    int j = unit_test[n];
    real x = predictor_test[n];
    real x_neg = fmin(x - k, 0.0);
    real x_pos = fmax(x - k, 0.0);
    real mu_n = alpha[j] + beta_neg[j] * x_neg + beta_pos[j] * x_pos;
    log_lik[n] = normal_lpdf(response_test[n] | mu_n, sigma[j]);
  }
}
