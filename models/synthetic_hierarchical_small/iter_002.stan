// Iteration 3: Student-t likelihood for robustness to outliers
// Looking at the data, some groups show high variance, t-distribution may help
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
  real<lower=0> sigma;   // within-group scale
  real<lower=1> nu;      // degrees of freedom for t-distribution
  vector[J] alpha;       // group means
}

model {
  // Priors
  mu ~ normal(0, 5);
  tau ~ normal(0, 2);
  sigma ~ normal(0, 2);
  nu ~ gamma(2, 0.1);    // encourages moderate nu (not too normal, not too heavy)

  // Hierarchical prior on group means
  alpha ~ normal(mu, tau);

  // Likelihood: Student-t for robustness
  effect_train ~ student_t(nu, alpha[unit_train], sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = student_t_lpdf(effect_test[n] | nu, alpha[unit_test[n]], sigma);
  }
}
