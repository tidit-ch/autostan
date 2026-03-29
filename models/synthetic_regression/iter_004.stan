// Iter 4: Correlated intercepts+slopes + unit-specific sigma (hierarchical heteroscedasticity)
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
  vector[2] mu;
  vector<lower=0>[2] tau;
  cholesky_factor_corr[2] L;
  matrix[2, J] z;
  // Hierarchical residual sigma
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[J] z_sigma;
}
transformed parameters {
  matrix[J, 2] ab;
  {
    matrix[2, 2] L_Sigma = diag_pre_multiply(tau, L);
    ab = (L_Sigma * z)';
  }
  vector[J] alpha = mu[1] + ab[, 1];
  vector[J] beta  = mu[2] + ab[, 2];
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
    response_train[n] ~ normal(alpha[unit_train[n]] + beta[unit_train[n]] * predictor_train[n], sigma[unit_train[n]]);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(response_test[n] | alpha[unit_test[n]] + beta[unit_test[n]] * predictor_test[n], sigma[unit_test[n]]);
  }
}
