// AutoStan: 1D Regression
// Iteration 1: Student-t likelihood + cubic polynomial mean
// Data shows heavy-tailed outliers (±12-17) and nonlinear mean.
// Student-t handles outliers; cubic polynomial captures nonlinear shape.

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
  real<lower=0> sigma;
  real<lower=1> nu;  // degrees of freedom for Student-t
}

model {
  alpha ~ normal(0, 5);
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 5);
  beta3 ~ normal(0, 5);
  sigma ~ normal(0, 5);
  nu ~ gamma(2, 0.1);  // prior favoring moderate-heavy tails

  vector[N_train] mu_train = alpha + beta1 * predictor_train
                             + beta2 * (predictor_train .* predictor_train)
                             + beta3 * (predictor_train .* predictor_train .* predictor_train);
  response_train ~ student_t(nu, mu_train, sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real mu_n = alpha + beta1 * predictor_test[n]
                + beta2 * (predictor_test[n]^2)
                + beta3 * (predictor_test[n]^3);
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma);
  }
}
