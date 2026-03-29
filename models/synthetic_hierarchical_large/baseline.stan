// Baseline: Simple hierarchical model, centered parameterization, shared sigma
data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> unit_train;
  array[N_test] int<lower=1,upper=J> unit_test;
  vector[N_train] effect_train;
  vector[N_test] effect_test;
}

parameters {
  real mu;               // global mean
  real<lower=0> tau;     // between-group SD
  real<lower=0> sigma;   // within-group SD (shared)
  vector[J] alpha;       // group means
}

model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 2);
  sigma ~ normal(0, 2);
  alpha ~ normal(mu, tau);
  effect_train ~ normal(alpha[unit_train], sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(effect_test[n] | alpha[unit_test[n]], sigma);
  }
}
