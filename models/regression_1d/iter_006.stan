// AutoStan: regression_1d iter 6
// Contamination mixture with Student-t clean component + sin(x) in mean.
// iter 5 NLPD=1.1244 (mixture + cubic + linear hetero + Normal clean component).
// Two improvements: (1) Student-t for clean component handles mild non-outlier deviations;
// (2) sin(x) added to mean since data pattern looks sinusoidal (peak ~x=1.5, trough ~x=4.7).

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
  real beta_sin;
  real log_sigma0;
  real log_sigma1;
  real<lower=0> sigma_out;
  real<lower=0, upper=1> pi_out;
  real<lower=2> nu;   // Student-t df for clean component
}

model {
  alpha      ~ normal(0, 5);
  beta1      ~ normal(0, 5);
  beta2      ~ normal(0, 5);
  beta3      ~ normal(0, 3);
  beta_sin   ~ normal(0, 3);
  log_sigma0 ~ normal(0, 2);
  log_sigma1 ~ normal(0, 1);
  sigma_out  ~ normal(0, 20);
  pi_out     ~ beta(1, 10);
  nu         ~ gamma(4, 0.5);   // prior mean 8, moderate tails

  for (n in 1:N_train) {
    real x = predictor_train[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3 + beta_sin*sin(x);
    real sigma_n = exp(log_sigma0 + log_sigma1*x);
    target += log_mix(pi_out,
                      normal_lpdf(response_train[n] | mu_n, sigma_out),
                      student_t_lpdf(response_train[n] | nu, mu_n, sigma_n));
  }
}

generated quantities {
  vector[N_test] log_lik;
  for (n in 1:N_test) {
    real x = predictor_test[n];
    real mu_n    = alpha + beta1*x + beta2*x^2 + beta3*x^3 + beta_sin*sin(x);
    real sigma_n = exp(log_sigma0 + log_sigma1*x);
    log_lik[n] = log_mix(pi_out,
                         normal_lpdf(response_test[n] | mu_n, sigma_out),
                         student_t_lpdf(response_test[n] | nu, mu_n, sigma_n));
  }
}
