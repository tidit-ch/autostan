// Iter 6: Simpler independent slopes+intercepts+quadratic + unit-specific sigma (no LKJ)
// Testing if LKJ correlation overhead is worth keeping given its marginal gain (iter1->iter2: 0.002)
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
  real mu_alpha;
  real mu_beta;
  real mu_gamma;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_gamma;
  vector[J] z_alpha;
  vector[J] z_beta;
  vector[J] z_gamma;
  // Hierarchical residual sigma
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[J] z_sigma;
}
transformed parameters {
  vector[J] alpha = mu_alpha + sigma_alpha * z_alpha;
  vector[J] beta  = mu_beta  + sigma_beta  * z_beta;
  vector[J] gamma = mu_gamma + sigma_gamma * z_gamma;
  vector<lower=0>[J] sigma = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}
model {
  mu_alpha ~ normal(0, 5);
  mu_beta  ~ normal(0, 5);
  mu_gamma ~ normal(0, 2);
  sigma_alpha ~ exponential(1);
  sigma_beta  ~ exponential(1);
  sigma_gamma ~ exponential(1);   // weaker regularization than iter 5's Exp(2)
  z_alpha ~ std_normal();
  z_beta  ~ std_normal();
  z_gamma ~ std_normal();
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
