// AutoStan: regression_1d iter 2
// Student-t + cubic polynomial mean + heteroscedastic noise (log-linear variance).
// iter 1 NLPD=1.5023 with quadratic+Student-t. Dataset explicitly has heteroscedastic noise.
// Adding cubic term for more mean flexibility; log-linear sigma(x) for heteroscedasticity.

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
  real beta3;
  real log_sigma0;       // intercept of log-sigma
  real log_sigma1;       // slope of log-sigma w.r.t. predictor
  real<lower=1> nu;
}

model {
  alpha       ~ normal(0, 5);
  beta1       ~ normal(0, 5);
  beta2       ~ normal(0, 5);
  beta3       ~ normal(0, 3);
  log_sigma0  ~ normal(0, 2);
  log_sigma1  ~ normal(0, 1);
  nu          ~ gamma(2, 0.1);

  for (n in 1:N_train) {
    real mu_n    = alpha + beta1 * predictor_train[n] + beta2 * predictor_train[n]^2 + beta3 * predictor_train[n]^3;
    real sigma_n = exp(log_sigma0 + log_sigma1 * predictor_train[n]);
    target += student_t_lpdf(response_train[n] | nu, mu_n, sigma_n);
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real mu_n    = alpha + beta1 * predictor_test[n] + beta2 * predictor_test[n]^2 + beta3 * predictor_test[n]^3;
    real sigma_n = exp(log_sigma0 + log_sigma1 * predictor_test[n]);
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma_n);
  }
}
