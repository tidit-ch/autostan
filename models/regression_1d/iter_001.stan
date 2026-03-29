// AutoStan: regression_1d iter 1
// Student-t likelihood (robust to outliers) + quadratic mean.
// Data shows 4 extreme outliers; normal errors inflate sigma, hurting all predictions.
// Nonlinear mean: response peaks around predictor ~1-2, declines at extremes.

data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  vector[N_train] predictor_train;
  vector[N_test] predictor_test;
  vector[N_train] response_train;
  vector[N_test] response_test;
}

parameters {
  real alpha;
  real beta1;
  real beta2;
  real<lower=0> sigma;
  real<lower=1> nu;   // degrees of freedom for Student-t
}

model {
  alpha  ~ normal(0, 5);
  beta1  ~ normal(0, 5);
  beta2  ~ normal(0, 5);
  sigma  ~ normal(0, 5);
  nu     ~ gamma(2, 0.1);  // prior: encourages moderate tail heaviness

  response_train ~ student_t(nu, alpha + beta1 * predictor_train + beta2 * predictor_train .* predictor_train, sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real mu_n = alpha + beta1 * predictor_test[n] + beta2 * predictor_test[n]^2;
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma);
  }
}
