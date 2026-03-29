// AutoStan: regression_1d iter 4
// Student-t + quartic polynomial mean + quadratic log-sigma heteroscedasticity.
// iter 2 NLPD=1.1558 (cubic+linear-hetero) was best; iter 3 Fourier was worse (1.2247).
// Adding quartic term for more mean flexibility; quadratic log-sigma allows variance
// to have a non-monotone shape (e.g., higher at extremes of predictor range).

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
  real beta4;
  real ls0;   // log-sigma intercept
  real ls1;   // log-sigma linear coef
  real ls2;   // log-sigma quadratic coef
  real<lower=1> nu;
}

model {
  alpha ~ normal(0, 5);
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 5);
  beta3 ~ normal(0, 3);
  beta4 ~ normal(0, 2);
  ls0   ~ normal(0, 2);
  ls1   ~ normal(0, 1);
  ls2   ~ normal(0, 1);
  nu    ~ gamma(2, 0.1);

  for (n in 1:N_train) {
    real x = predictor_train[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3 + beta4*x^4;
    real sigma_n = exp(ls0 + ls1*x + ls2*x^2);
    target += student_t_lpdf(response_train[n] | nu, mu_n, sigma_n);
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3 + beta4*x^4;
    real sigma_n = exp(ls0 + ls1*x + ls2*x^2);
    log_lik[n] = student_t_lpdf(response_test[n] | nu, mu_n, sigma_n);
  }
}
