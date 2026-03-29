// Iteration 2: Non-centered parameterization for better sampling
// Uses z-scores for group effects to improve geometry
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
  vector[J] z_alpha;     // non-centered group effects
}

transformed parameters {
  vector[J] alpha = mu + tau * z_alpha;
}

model {
  // Priors
  mu ~ normal(0, 5);
  tau ~ normal(0, 2);
  sigma ~ normal(0, 2);

  // Non-centered parameterization
  z_alpha ~ std_normal();

  // Likelihood
  effect_train ~ normal(alpha[unit_train], sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(effect_test[n] | alpha[unit_test[n]], sigma);
  }
}
