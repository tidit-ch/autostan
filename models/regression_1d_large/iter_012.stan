// AutoStan: 1D Regression
// Iteration 12: Mixture (sigma_out=10) + learnable omega + sine+linear log-sigma
// Iter 11 NLPD=1.2256 used cubic polynomial log-sigma.
// The variance may follow the same sinusoidal structure as the mean
// (larger noise in high-response region x~1.5, smaller at x~0 and x~3).
// Trying log(sigma) = s0 + s1*sin(omega*x) + s2*cos(omega*x) + s3*x.

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
  real s_sin;   // sine component of log-sigma
  real s_cos;   // cosine component of log-sigma
  real s_lin;   // linear component of log-sigma
  real<lower=0, upper=1> pi_out;
}

model {
  alpha ~ normal(0, 5);
  a1 ~ normal(0, 5);
  b1 ~ normal(0, 5);
  beta_lin ~ normal(0, 5);
  omega ~ normal(pi() / 3.0, 0.3);
  s0 ~ normal(0, 2);
  s_sin ~ normal(0, 1);
  s_cos ~ normal(0, 1);
  s_lin ~ normal(0, 0.5);
  pi_out ~ beta(1, 9);

  vector[N_train] mu_train = alpha
    + a1 * sin(omega * predictor_train)
    + b1 * cos(omega * predictor_train)
    + beta_lin * predictor_train;
  vector[N_train] sigma_train = exp(s0
    + s_sin * sin(omega * predictor_train)
    + s_cos * cos(omega * predictor_train)
    + s_lin * predictor_train);

  for (n in 1:N_train) {
    target += log_mix(pi_out,
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
    real sigma_n = exp(s0
      + s_sin * sin(omega * x)
      + s_cos * cos(omega * x)
      + s_lin * x);
    log_lik[n] = log_mix(pi_out,
                         normal_lpdf(response_test[n] | mu_n, sigma_out),
                         normal_lpdf(response_test[n] | mu_n, sigma_n));
  }
}
