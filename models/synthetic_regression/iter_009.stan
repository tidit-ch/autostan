// Iter 9: 3D LKJ(4) - stronger regularization toward independence for 15-group correlation matrix
// Keeping iter 7's Exp priors on tau which worked best
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
  vector[3] mu;
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
  vector[J] alpha = mu[1] + abg[, 1];
  vector[J] beta  = mu[2] + abg[, 2];
  vector[J] gamma = mu[3] + abg[, 3];
  vector<lower=0>[J] sigma = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}
model {
  mu[1] ~ normal(0, 5);
  mu[2] ~ normal(0, 5);
  mu[3] ~ normal(0, 2);
  tau[1] ~ exponential(1);
  tau[2] ~ exponential(1);
  tau[3] ~ exponential(2);   // iter 7 priors
  L ~ lkj_corr_cholesky(4);  // LKJ(4) vs LKJ(2) in iter 7
  to_vector(z) ~ std_normal();
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ exponential(1);
  z_sigma ~ std_normal();
  for (n in 1:N_train) {
    int j = unit_train[n];
    real mu_n = alpha[j] + beta[j] * predictor_train[n] + gamma[j] * predictor_train[n]^2;
    response_train[n] ~ normal(mu_n, sigma[j]);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    int j = unit_test[n];
    real mu_n = alpha[j] + beta[j] * predictor_test[n] + gamma[j] * predictor_test[n]^2;
    log_lik[n] = normal_lpdf(response_test[n] | mu_n, sigma[j]);
  }
}
