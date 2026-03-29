// Iteration 1: Baseline hierarchical normal model
// Partial pooling: group means drawn from a common prior
// Within-group variance estimated globally
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
  real<lower=0> sigma;   // within-group SD
  vector[J] alpha;       // group means (non-centered)
}

model {
  // Priors
  mu ~ normal(0, 5);
  tau ~ normal(0, 2);
  sigma ~ normal(0, 2);

  // Hierarchical prior on group means
  alpha ~ normal(mu, tau);

  // Likelihood
  effect_train ~ normal(alpha[unit_train], sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(effect_test[n] | alpha[unit_test[n]], sigma);
  }
}
