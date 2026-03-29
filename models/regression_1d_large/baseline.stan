// AutoStan: 1D Regression baseline
// Iteration 0: Simple linear regression with Normal likelihood
// Deliberately simple to establish baseline NLPD.

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
  real beta;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sigma ~ normal(0, 5);

  response_train ~ normal(alpha + beta * predictor_train, sigma);
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    log_lik[n] = normal_lpdf(response_test[n] | alpha + beta * predictor_test[n], sigma);
  }
}
