// Iteration 4: NCP + heteroscedastic + Student-t + tighter priors informed by data
// tau ~ half-normal(0,1) since between-group SD ~0.86; tau_sigma tighter ~0.3
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
  real<lower=2> nu;           // degrees of freedom for t-likelihood
}

transformed parameters {
  vector[J] alpha = mu + tau * alpha_raw;
  vector<lower=0>[J] sigma_j = exp(mu_log_sigma + tau_sigma * log_sigma_raw);
}

model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 1);          // tighter: data shows tau~0.86
  mu_log_sigma ~ normal(0, 1); // global log-sigma (data sigma~1.0, log~0)
  tau_sigma ~ normal(0, 0.3);  // tighter: within-group SDs 0.84-1.21, log-SD variation ~0.17
  alpha_raw ~ std_normal();
  log_sigma_raw ~ std_normal();
  nu ~ gamma(2, 0.1);
  effect_train ~ student_t(nu, alpha[unit_train], sigma_j[unit_train]);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = student_t_lpdf(effect_test[n] | nu, alpha[unit_test[n]], sigma_j[unit_test[n]]);
  }
}
