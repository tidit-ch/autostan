// Iteration 5: NCP + heteroscedastic + Student-t nu=10 fixed + tighter priors
// Data is approximately normal; fixing nu=10 gives mild robustness without nu uncertainty
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
  vector[J] alpha_raw;        // group mean offsets (NCP)
  vector[J] log_sigma_raw;    // group-specific log SDs (NCP)
}

transformed parameters {
  vector[J] alpha = mu + tau * alpha_raw;
  vector<lower=0>[J] sigma_j = exp(mu_log_sigma + tau_sigma * log_sigma_raw);
}

model {
  mu ~ normal(0, 2);
  tau ~ normal(0, 1);
  mu_log_sigma ~ normal(0, 1);
  tau_sigma ~ normal(0, 0.3);
  alpha_raw ~ std_normal();
  log_sigma_raw ~ std_normal();
  // Student-t with fixed nu=10: mild robustness without extra parameter uncertainty
  for (n in 1:N_train) {
    effect_train[n] ~ student_t(10, alpha[unit_train[n]], sigma_j[unit_train[n]]);
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = student_t_lpdf(effect_test[n] | 10, alpha[unit_test[n]], sigma_j[unit_test[n]]);
  }
}
