// Iter 10: Hierarchical piecewise linear (hinge at x=0) + unit-specific sigma
// Different slopes for x<0 and x>0 per unit, correlated via 3D LKJ(2)
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
  // Piecewise basis: x_neg = min(x,0), x_pos = max(x,0)
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
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[J] z_sigma;
}
transformed parameters {
  matrix[J, 3] abg;
  {
    matrix[3, 3] L_Sigma = diag_pre_multiply(tau, L);
    abg = (L_Sigma * z)';
  }
  vector[J] alpha    = mu[1] + abg[, 1];
  vector[J] beta_neg = mu[2] + abg[, 2];   // slope for x <= 0
  vector[J] beta_pos = mu[3] + abg[, 3];   // slope for x >= 0
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
  for (n in 1:N_train) {
    int j = unit_train[n];
    real mu_n = alpha[j] + beta_neg[j] * x_neg_train[n] + beta_pos[j] * x_pos_train[n];
    response_train[n] ~ normal(mu_n, sigma[j]);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    int j = unit_test[n];
    real mu_n = alpha[j] + beta_neg[j] * x_neg_test[n] + beta_pos[j] * x_pos_test[n];
    log_lik[n] = normal_lpdf(response_test[n] | mu_n, sigma[j]);
  }
}
