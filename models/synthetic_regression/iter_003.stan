// Iter 3: Correlated intercepts+slopes + Student-t likelihood for robustness
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
  vector[2] mu;
  vector<lower=0>[2] tau;
  cholesky_factor_corr[2] L;
  matrix[2, J] z;
  real<lower=0> sigma;
  real<lower=1> nu;               // degrees of freedom for Student-t
}
transformed parameters {
  matrix[J, 2] ab;
  {
    matrix[2, 2] L_Sigma = diag_pre_multiply(tau, L);
    ab = (L_Sigma * z)';
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
  nu ~ gamma(2, 0.1);             // prior peaks around 10-20 dof, heavy tail allows small nu
  for (n in 1:N_train) {
    response_train[n] ~ student_t(nu, alpha[unit_train[n]] + beta[unit_train[n]] * predictor_train[n], sigma);
  }
}
generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = student_t_lpdf(response_test[n] | nu, alpha[unit_test[n]] + beta[unit_test[n]] * predictor_test[n], sigma);
  }
}
