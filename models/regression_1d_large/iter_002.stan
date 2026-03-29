// AutoStan: 1D Regression
// Iteration 2: Student-t likelihood + sine basis mean
// The data shows an arch shape: ~0 at x=0, peak at x~1.5, ~0 at x=3, negative for x>3.
// sin(pi*x/3) captures this naturally. Adding linear term for flexibility.
// Keeping Student-t from iter 1 (NLPD=1.3181) for outlier robustness.

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
  real beta_sin;   // coefficient for sin(pi*x/3)
  real beta_cos;   // coefficient for cos(pi*x/3) — adds phase flexibility
  real beta_lin;   // linear trend term
  real<lower=0> sigma;
  real<lower=1> nu;
}

model {
  alpha ~ normal(0, 5);
  beta_sin ~ normal(0, 5);
  beta_cos ~ normal(0, 5);
  beta_lin ~ normal(0, 5);
  sigma ~ normal(0, 5);
  nu ~ gamma(2, 0.1);

  vector[N_train] mu_train = alpha
                             + beta_sin * sin(pi() / 3.0 * predictor_train)
                             + beta_cos * cos(pi() / 3.0 * predictor_train)
                             + beta_lin * predictor_train;
  response_train ~ student_t(nu, mu_train, sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real mu_n = alpha
                + beta_sin * sin(pi() / 3.0 * predictor_test[n])
                + beta_cos * cos(pi() / 3.0 * predictor_test[n])
                + beta_lin * predictor_test[n];
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma);
  }
}
