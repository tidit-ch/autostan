// AutoStan: 1D Regression
// Iteration 3: Student-t + 2-harmonic Fourier basis + linear
// Iter 2 (sine basis, NLPD=1.3060) improved over cubic polynomial.
// Adding 2nd harmonic to capture finer structure in the arch shape.

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
  real a1;  // sin(pi*x/3)
  real b1;  // cos(pi*x/3)
  real a2;  // sin(2*pi*x/3)
  real b2;  // cos(2*pi*x/3)
  real beta_lin;
  real<lower=0> sigma;
  real<lower=1> nu;
}

model {
  alpha ~ normal(0, 5);
  a1 ~ normal(0, 5);
  b1 ~ normal(0, 5);
  a2 ~ normal(0, 5);
  b2 ~ normal(0, 5);
  beta_lin ~ normal(0, 5);
  sigma ~ normal(0, 5);
  nu ~ gamma(2, 0.1);

  real omega = pi() / 3.0;
  vector[N_train] mu_train = alpha
    + a1 * sin(omega * predictor_train)
    + b1 * cos(omega * predictor_train)
    + a2 * sin(2 * omega * predictor_train)
    + b2 * cos(2 * omega * predictor_train)
    + beta_lin * predictor_train;
  response_train ~ student_t(nu, mu_train, sigma);
}

generated quantities {
  vector[N_test] log_lik;
  real omega = pi() / 3.0;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n = alpha
      + a1 * sin(omega * x)
      + b1 * cos(omega * x)
      + a2 * sin(2 * omega * x)
      + b2 * cos(2 * omega * x)
      + beta_lin * x;
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma);
  }
}
