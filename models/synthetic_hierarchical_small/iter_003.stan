// Iteration 4: Hierarchical model with group-specific within-group variances
// Each group has its own sigma, drawn from a common log-normal prior
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
  real mu;                    // global mean
  real<lower=0> tau;          // between-group SD for means
  real mu_log_sigma;          // global log within-group SD
  real<lower=0> tau_sigma;    // between-group SD for log sigmas
  vector[J] alpha;            // group means
  vector[J] log_sigma_raw;    // group-specific log SDs (non-centered)
}

transformed parameters {
  vector<lower=0>[J] sigma_j = exp(mu_log_sigma + tau_sigma * log_sigma_raw);
}

model {
  // Priors
  mu ~ normal(0, 5);
  tau ~ normal(0, 2);
  mu_log_sigma ~ normal(0, 1);
  tau_sigma ~ normal(0, 0.5);

  // Hierarchical priors
  alpha ~ normal(mu, tau);
  log_sigma_raw ~ std_normal();

  // Likelihood
  for (n in 1:N_train) {
    effect_train[n] ~ normal(alpha[unit_train[n]], sigma_j[unit_train[n]]);
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(effect_test[n] | alpha[unit_test[n]], sigma_j[unit_test[n]]);
  }
}
