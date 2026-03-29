// AutoStan: 1D Regression
// Iteration 13: Mixture + learnable omega + cubic log-sigma + spatially-varying pi_out
// Iter 11 NLPD=1.2256 (best). Iter 12 sine log-sigma didn't help.
// Now keeping cubic log-sigma from iter 11 and adding spatially-varying outlier probability:
// logit(pi_out(x)) = p0 + p1*x — some x-regions may have systematically more outliers.

data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  vector[N_train] predictor_train;
  vector[N_test] predictor_test;
  vector[N_train] response_train;
  vector[N_test] response_test;
}

transformed data {
  real sigma_out = 10.0;
}

parameters {
  real alpha;
  real a1;
  real b1;
  real beta_lin;
  real<lower=0.3, upper=3.0> omega;
  real s0;
  real s1;
  real s2;
  real s3;
  real p0;  // logit outlier probability intercept
  real p1;  // logit outlier probability slope
}

model {
  alpha ~ normal(0, 5);
  a1 ~ normal(0, 5);
  b1 ~ normal(0, 5);
  beta_lin ~ normal(0, 5);
  omega ~ normal(pi() / 3.0, 0.3);
  s0 ~ normal(0, 2);
  s1 ~ normal(0, 1);
  s2 ~ normal(0, 0.5);
  s3 ~ normal(0, 0.2);
  p0 ~ normal(-2.2, 1);  // centered at logit(0.1) ≈ -2.2
  p1 ~ normal(0, 0.5);

  vector[N_train] x2 = predictor_train .* predictor_train;
  vector[N_train] x3 = x2 .* predictor_train;
  vector[N_train] mu_train = alpha
    + a1 * sin(omega * predictor_train)
    + b1 * cos(omega * predictor_train)
    + beta_lin * predictor_train;
  vector[N_train] sigma_train = exp(s0 + s1 * predictor_train + s2 * x2 + s3 * x3);
  vector[N_train] pi_train = inv_logit(p0 + p1 * predictor_train);

  for (n in 1:N_train) {
    target += log_mix(pi_train[n],
                      normal_lpdf(response_train[n] | mu_train[n], sigma_out),
                      normal_lpdf(response_train[n] | mu_train[n], sigma_train[n]));
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n = alpha
      + a1 * sin(omega * x)
      + b1 * cos(omega * x)
      + beta_lin * x;
    real sigma_n = exp(s0 + s1 * x + s2 * x^2 + s3 * x^3);
    real pi_n = inv_logit(p0 + p1 * x);
    log_lik[n] = log_mix(pi_n,
                         normal_lpdf(response_test[n] | mu_n, sigma_out),
                         normal_lpdf(response_test[n] | mu_n, sigma_n));
  }
}
