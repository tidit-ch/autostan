// AutoStan: 1D Regression
// Iteration 6: Student-t + sine+quadratic mean + quadratic log-sigma
// Iter 5 (sine + quadratic het., NLPD=1.2854). Now adding quadratic polynomial
// correction to sine basis — may capture residual structure in the mean.

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
  real a1;
  real b1;
  real beta_lin;
  real beta_quad;
  real s0;
  real s1;
  real s2;
  real<lower=1> nu;
}

model {
  alpha ~ normal(0, 5);
  a1 ~ normal(0, 5);
  b1 ~ normal(0, 5);
  beta_lin ~ normal(0, 5);
  beta_quad ~ normal(0, 2);
  s0 ~ normal(0, 2);
  s1 ~ normal(0, 1);
  s2 ~ normal(0, 0.5);
  nu ~ gamma(2, 0.1);

  real omega = pi() / 3.0;
  vector[N_train] x2 = predictor_train .* predictor_train;
  vector[N_train] mu_train = alpha
    + a1 * sin(omega * predictor_train)
    + b1 * cos(omega * predictor_train)
    + beta_lin * predictor_train
    + beta_quad * x2;
  vector[N_train] sigma_train = exp(s0 + s1 * predictor_train + s2 * x2);

  for (n in 1:N_train) {
    response_train[n] ~ student_t(nu, mu_train[n], sigma_train[n]);
  }
}

generated quantities {
  vector[N_test] log_lik;
  real omega = pi() / 3.0;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n = alpha
      + a1 * sin(omega * x)
      + b1 * cos(omega * x)
      + beta_lin * x
      + beta_quad * x^2;
    real sigma_n = exp(s0 + s1 * x + s2 * x^2);
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma_n);
  }
}
