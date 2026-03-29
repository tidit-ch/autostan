// AutoStan: 1D Regression
// Iteration 14: Student-t inlier + Normal outlier mixture + learnable omega + cubic log-sigma
// Iter 11 NLPD=1.2256 (best). Iters 12,13 didn't improve (streak=2, last chance).
// Replacing Normal inlier with Student-t inlier to handle moderate deviations within inlier class.
// Extreme outliers (±12-17) still handled by Normal(mu, 10) outlier component.

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
  real<lower=0, upper=1> pi_out;
  real<lower=2> nu;  // degrees of freedom for inlier Student-t
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
  pi_out ~ beta(1, 9);
  nu ~ gamma(4, 0.3);  // mode at ~10, allows moderate tails

  vector[N_train] x2 = predictor_train .* predictor_train;
  vector[N_train] x3 = x2 .* predictor_train;
  vector[N_train] mu_train = alpha
    + a1 * sin(omega * predictor_train)
    + b1 * cos(omega * predictor_train)
    + beta_lin * predictor_train;
  vector[N_train] sigma_train = exp(s0 + s1 * predictor_train + s2 * x2 + s3 * x3);

  for (n in 1:N_train) {
    target += log_mix(pi_out,
                      normal_lpdf(response_train[n] | mu_train[n], sigma_out),
                      student_t_lpdf(response_train[n] | nu, mu_train[n], sigma_train[n]));
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
    log_lik[n] = log_mix(pi_out,
                         normal_lpdf(response_test[n] | mu_n, sigma_out),
                         student_t_lpdf(response_test[n] | nu, mu_n, sigma_n));
  }
}
