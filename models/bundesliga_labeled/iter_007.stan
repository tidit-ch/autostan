// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 7: Asymmetric likelihood — NegBin for home goals, Poisson for away.
// Training data: home var/mean = 2.205/1.754 = 1.26 (overdispersed),
//                away var/mean = 1.471/1.420 = 1.04 (nearly Poisson).
// Method-of-moments estimate: phi_home ≈ 1.754/(2.205-1.754) ≈ 3.9.
// Using NB only for home avoids the instability from iter 2 (single phi both).
// Base: iter 6 NCP hierarchical Poisson (best NLPD=1.5460).

data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> J;
  array[N_train] int<lower=1,upper=J> home_train;
  array[N_train] int<lower=1,upper=J> away_train;
  array[N_train] int<lower=0> goals_home_train;
  array[N_train] int<lower=0> goals_away_train;
  array[N_test] int<lower=1,upper=J> home_test;
  array[N_test] int<lower=1,upper=J> away_test;
  array[N_test] int<lower=0> goals_home_test;
  array[N_test] int<lower=0> goals_away_test;
}

parameters {
  real home_adv;
  real mu;
  real<lower=0> sigma_att;
  real<lower=0> sigma_def;
  vector[J] attack_raw;
  vector[J] defense_raw;
  real<lower=0> phi_home;  // NB dispersion for home goals; large phi -> Poisson
}

transformed parameters {
  vector[J] attack  = sigma_att * attack_raw;
  vector[J] defense = sigma_def * defense_raw;
}

model {
  home_adv ~ normal(0.2, 0.5);
  mu ~ normal(0.35, 0.5);
  sigma_att ~ normal(0, 0.5);
  sigma_def ~ normal(0, 0.5);
  attack_raw  ~ normal(0, 1);
  defense_raw ~ normal(0, 1);
  phi_home ~ normal(4, 3);  // centered on MOM estimate ~3.9

  for (n in 1:N_train) {
    real lambda_home = exp(mu + home_adv + attack[home_train[n]] - defense[away_train[n]]);
    real lambda_away = exp(mu + attack[away_train[n]] - defense[home_train[n]]);
    goals_home_train[n] ~ neg_binomial_2(lambda_home, phi_home);
    goals_away_train[n] ~ poisson(lambda_away);
  }
}

generated quantities {
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lambda_home = exp(mu + home_adv + attack[home_test[n]] - defense[away_test[n]]);
    real lambda_away = exp(mu + attack[away_test[n]] - defense[home_test[n]]);
    log_lik[n] = neg_binomial_2_lpmf(goals_home_test[n] | lambda_home, phi_home);
    log_lik[N_test + n] = poisson_lpmf(goals_away_test[n] | lambda_away);
  }
}
