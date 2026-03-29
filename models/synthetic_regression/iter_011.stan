// Iter 11: 3-segment piecewise linear (knots at x=-1 and x=+1) + unit-specific sigma, 4D LKJ
data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> unit_train;
  array[N_test] int<lower=1,upper=J> unit_test;
  vector[N_train] predictor_train;
  vector[N_test] predictor_test;
  vector[N_train] response_train;
  vector[N_test] response_test;
}
transformed data {
  // Hinge basis: left (x<-1), middle (-1<=x<=1), right (x>1)
  // Using ReLU-like basis: h1 = max(x - (-1), 0), h2 = max(x - 1, 0)
  // Then: f(x) = alpha + beta_left*x + delta1*h1 + delta2*h2
  // This represents 3 piecewise linear segments with knots at -1 and 1
  vector[N_train] x_train = predictor_train;
  vector[N_train] h1_train; // kink at -1
  vector[N_train] h2_train; // kink at 1
  vector[N_test]  x_test  = predictor_test;
  vector[N_test]  h1_test;
  vector[N_test]  h2_test;
  for (n in 1:N_train) {
    h1_train[n] = fmax(predictor_train[n] + 1.0, 0.0);  // max(x+1, 0): ramp starting at x=-1
    h2_train[n] = fmax(predictor_train[n] - 1.0, 0.0);  // max(x-1, 0): ramp starting at x=+1
  }
  for (n in 1:N_test) {
    h1_test[n] = fmax(predictor_test[n] + 1.0, 0.0);
    h2_test[n] = fmax(predictor_test[n] - 1.0, 0.0);
  }
}
parameters {
  vector[4] mu;                    // [mu_alpha, mu_beta, mu_delta1, mu_delta2]
  vector<lower=0>[4] tau;
  cholesky_factor_corr[4] L;
  matrix[4, J] z;
  real mu_log_sigma;
  real<lower=0> sigma_log_sigma;
  vector[J] z_sigma;
}
transformed parameters {
  matrix[J, 4] params;
  {
    matrix[4, 4] L_Sigma = diag_pre_multiply(tau, L);
    params = (L_Sigma * z)';
  }
  vector[J] alpha  = mu[1] + params[, 1];
  vector[J] beta   = mu[2] + params[, 2];   // base slope (for x < -1 region)
  vector[J] delta1 = mu[3] + params[, 3];   // slope change at x=-1
  vector[J] delta2 = mu[4] + params[, 4];   // slope change at x=+1
  vector<lower=0>[J] sigma = exp(mu_log_sigma + sigma_log_sigma * z_sigma);
}
model {
  mu[1] ~ normal(0, 5);
  mu[2] ~ normal(0, 5);
  mu[3] ~ normal(0, 2);   // kink magnitudes expected moderate
  mu[4] ~ normal(0, 2);
  tau[1] ~ exponential(1);
  tau[2] ~ exponential(1);
  tau[3] ~ exponential(1);
  tau[4] ~ exponential(1);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  mu_log_sigma ~ normal(0, 1);
  sigma_log_sigma ~ exponential(1);
  z_sigma ~ std_normal();
  for (n in 1:N_train) {
    int j = unit_train[n];
    real mu_n = alpha[j] + beta[j] * x_train[n] + delta1[j] * h1_train[n] + delta2[j] * h2_train[n];
    response_train[n] ~ normal(mu_n, sigma[j]);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    int j = unit_test[n];
    real mu_n = alpha[j] + beta[j] * x_test[n] + delta1[j] * h1_test[n] + delta2[j] * h2_test[n];
    log_lik[n] = normal_lpdf(response_test[n] | mu_n, sigma[j]);
  }
}
