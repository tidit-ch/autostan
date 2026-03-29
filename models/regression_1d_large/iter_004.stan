// AutoStan: 1D Regression
// Iteration 4: Student-t + sine basis (iter 2 best) + heteroscedastic sigma
// Dataset is described as "heteroscedastic noise, outliers, nonlinear mean."
// Iter 2 (sine+linear, NLPD=1.3060) handles nonlinear mean and outliers.
// Now modeling log(sigma) as a linear function of x to capture heteroscedasticity.

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
  real s0;       // log-sigma intercept
  real s1;       // log-sigma slope in x
  real<lower=1> nu;
}

model {
  alpha ~ normal(0, 5);
  a1 ~ normal(0, 5);
  b1 ~ normal(0, 5);
  beta_lin ~ normal(0, 5);
  s0 ~ normal(0, 2);
  s1 ~ normal(0, 1);
  nu ~ gamma(2, 0.1);

  real omega = pi() / 3.0;
  vector[N_train] mu_train = alpha
    + a1 * sin(omega * predictor_train)
    + b1 * cos(omega * predictor_train)
    + beta_lin * predictor_train;
  vector[N_train] sigma_train = exp(s0 + s1 * predictor_train);

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
      + beta_lin * x;
    real sigma_n = exp(s0 + s1 * x);
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma_n);
  }
}
