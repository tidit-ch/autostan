// Iter 1: Hierarchical varying intercepts and slopes (non-centered parameterization)
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
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  real<lower=0> sigma;
  vector[J] z_alpha;
  vector[J] z_beta;
}
transformed parameters {
  vector[J] alpha = mu_alpha + sigma_alpha * z_alpha;
  vector[J] beta  = mu_beta  + sigma_beta  * z_beta;
}
model {
  mu_alpha ~ normal(0, 5);
  mu_beta  ~ normal(0, 5);
  sigma_alpha ~ exponential(1);
  sigma_beta  ~ exponential(1);
  sigma ~ exponential(1);
  z_alpha ~ std_normal();
  z_beta  ~ std_normal();
  response_train ~ normal(alpha[unit_train] + beta[unit_train] .* predictor_train, sigma);
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(response_test[n] | alpha[unit_test[n]] + beta[unit_test[n]] * predictor_test[n], sigma);
  }
}
