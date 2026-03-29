// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 0 (baseline): Simple Poisson model with per-team attack/defense
// and home advantage. Independent normal priors.

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
  vector[J] attack;
  vector[J] defense;
}

model {
  home_adv ~ normal(0, 1);
  mu ~ normal(0, 1);
  attack ~ normal(0, 1);
  defense ~ normal(0, 1);

  for (n in 1:N_train) {
    real lambda_home = exp(mu + home_adv + attack[home_train[n]] - defense[away_train[n]]);
    real lambda_away = exp(mu + attack[away_train[n]] - defense[home_train[n]]);
    goals_home_train[n] ~ poisson(lambda_home);
    goals_away_train[n] ~ poisson(lambda_away);
  }
}

generated quantities {
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lambda_home = exp(mu + home_adv + attack[home_test[n]] - defense[away_test[n]]);
    real lambda_away = exp(mu + attack[away_test[n]] - defense[home_test[n]]);
    log_lik[n] = poisson_lpmf(goals_home_test[n] | lambda_home);
    log_lik[N_test + n] = poisson_lpmf(goals_away_test[n] | lambda_away);
  }
}
