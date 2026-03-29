// AutoStan: regression_1d iter 8
// Contamination mixture + quartic polynomial mean + linear log-sigma.
// iter 5 NLPD=1.1244 best (cubic+mixture); iters 6,7 non-improving.
// iter 4 tested quartic WITHOUT mixture (NLPD=1.2319); now combining quartic with mixture.
// Quartic allows an extra inflection point vs cubic; mixture handles outliers.
// Centering predictor at mean (~2.0) reduces polynomial collinearity.

data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  vector[N_train] predictor_train;
  vector[N_test] predictor_test;
  vector[N_train] response_train;
  vector[N_test] response_test;
}

transformed data {
  real x_center = 2.0;   // approximate mean of predictor
  vector[N_train] xc_train = predictor_train - x_center;
  vector[N_test]  xc_test  = predictor_test  - x_center;
}

parameters {
  real alpha;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
  real log_sigma0;
  real log_sigma1;
  real<lower=0> sigma_out;
  real<lower=0, upper=1> pi_out;
}

model {
  alpha      ~ normal(0, 5);
  beta1      ~ normal(0, 5);
  beta2      ~ normal(0, 5);
  beta3      ~ normal(0, 3);
  beta4      ~ normal(0, 2);
  log_sigma0 ~ normal(0, 2);
  log_sigma1 ~ normal(0, 1);
  sigma_out  ~ normal(0, 20);
  pi_out     ~ beta(1, 10);

  for (n in 1:N_train) {
    real x = xc_train[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3 + beta4*x^4;
    real sigma_n = exp(log_sigma0 + log_sigma1*predictor_train[n]);
    target += log_mix(pi_out,
                      normal_lpdf(response_train[n] | mu_n, sigma_out),
                      normal_lpdf(response_train[n] | mu_n, sigma_n));
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real x = xc_test[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3 + beta4*x^4;
    real sigma_n = exp(log_sigma0 + log_sigma1*predictor_test[n]);
    log_lik[n] = log_mix(pi_out,
                         normal_lpdf(response_test[n] | mu_n, sigma_out),
                         normal_lpdf(response_test[n] | mu_n, sigma_n));
  }
}
