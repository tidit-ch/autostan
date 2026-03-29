// AutoStan: regression_1d iter 3
// Student-t + heteroscedastic + Fourier basis mean (sin/cos terms).
// iter 2 NLPD=1.1558 with cubic+hetero+t. Data pattern looks oscillatory.
// Fourier basis (sin(x), cos(x), sin(2x), cos(2x)) can capture periodic/nonlinear patterns
// more flexibly than polynomials which can explode at extremes.

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
  real a1;   // sin(x)
  real b1;   // cos(x)
  real a2;   // sin(2x)
  real b2;   // cos(2x)
  real log_sigma0;
  real log_sigma1;
  real<lower=1> nu;
}

model {
  alpha ~ normal(0, 5);
  a1 ~ normal(0, 3);
  b1 ~ normal(0, 3);
  a2 ~ normal(0, 3);
  b2 ~ normal(0, 3);
  log_sigma0 ~ normal(0, 2);
  log_sigma1 ~ normal(0, 1);
  nu ~ gamma(2, 0.1);

  for (n in 1:N_train) {
    real x = predictor_train[n];
    real mu_n = alpha + a1 * sin(x) + b1 * cos(x) + a2 * sin(2*x) + b2 * cos(2*x);
    real sigma_n = exp(log_sigma0 + log_sigma1 * x);
    target += student_t_lpdf(response_train[n] | nu, mu_n, sigma_n);
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n = alpha + a1 * sin(x) + b1 * cos(x) + a2 * sin(2*x) + b2 * cos(2*x);
    real sigma_n = exp(log_sigma0 + log_sigma1 * x);
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma_n);
  }
}
