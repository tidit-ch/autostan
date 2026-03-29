// Iter 2: Correlated varying intercepts and slopes via Cholesky (LKJ prior)
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
parameters {
  vector[2] mu;                    // [mu_alpha, mu_beta]
  vector<lower=0>[2] tau;          // [sigma_alpha, sigma_beta]
  cholesky_factor_corr[2] L;       // Cholesky of correlation matrix
  matrix[2, J] z;                  // non-centered group offsets
  real<lower=0> sigma;
}
transformed parameters {
  matrix[J, 2] ab;
  {
    matrix[2, 2] L_Sigma = diag_pre_multiply(tau, L);
    ab = (L_Sigma * z)';  // J x 2 matrix: [alpha_j, beta_j]
  }
  vector[J] alpha = mu[1] + ab[, 1];
  vector[J] beta  = mu[2] + ab[, 2];
}
model {
  mu ~ normal(0, 5);
  tau ~ exponential(1);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  sigma ~ exponential(1);
  response_train ~ normal(alpha[unit_train] + beta[unit_train] .* predictor_train, sigma);
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(response_test[n] | alpha[unit_test[n]] + beta[unit_test[n]] * predictor_test[n], sigma);
  }
}
