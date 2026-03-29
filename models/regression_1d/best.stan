// AutoStan: regression_1d iter 5
// Gaussian contamination mixture + cubic polynomial + linear log-sigma.
// iter 2 NLPD=1.1558 (cubic+linear-hetero+Student-t) was best.
// 4 extreme outliers (|y| > 11) are very far from mean; explicit contamination model
// may give better predictive distributions than Student-t:
// y_n ~ (1-pi)*Normal(mu_n, sigma_n) + pi*Normal(mu_n, sigma_out)
// where sigma_out >> sigma_n, capturing the outlier process explicitly.

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
  real log_sigma0;
  real log_sigma1;
  real<lower=0> sigma_out;   // outlier scale
  real<lower=0, upper=1> pi_out;   // contamination probability
}

model {
  alpha      ~ normal(0, 5);
  beta1      ~ normal(0, 5);
  beta2      ~ normal(0, 5);
  beta3      ~ normal(0, 3);
  log_sigma0 ~ normal(0, 2);
  log_sigma1 ~ normal(0, 1);
  sigma_out  ~ normal(0, 20);
  pi_out     ~ beta(1, 10);   // prior mean ~0.09; expect ~5-10% outliers

  for (n in 1:N_train) {
    real x = predictor_train[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3;
    real sigma_n = exp(log_sigma0 + log_sigma1*x);
    target += log_mix(pi_out,
                      normal_lpdf(response_train[n] | mu_n, sigma_out),
                      normal_lpdf(response_train[n] | mu_n, sigma_n));
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3;
    real sigma_n = exp(log_sigma0 + log_sigma1*x);
    log_lik[n] = log_mix(pi_out,
                         normal_lpdf(response_test[n] | mu_n, sigma_out),
                         normal_lpdf(response_test[n] | mu_n, sigma_n));
  }
}
