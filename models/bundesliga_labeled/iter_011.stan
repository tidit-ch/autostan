// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 11: Separate defensive home advantage delta_def[h], independent of delta_att[h].
// lambda_home = exp(mu + delta_att[h] + attack[h] - defense[a])  [iter 9 offensive side]
// lambda_away = exp(mu + attack[a] - defense[h] - delta_def[h])  [NEW: home defense advantage]
// sigma_dd ~ N+(0, 0.15): very tight — only activates if data supports home defense effect.
// Iter 10 used single delta for both sides (symmetric): NLPD=1.5443 (worse than iter 9).
// Separate parameters allow asymmetric home effects per team.
// Base: iter 9 (best NLPD=1.5432).

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
  real delta_att_mu;              // global offensive home advantage
  real delta_def_mu;              // global defensive home advantage
  real<lower=0> sigma_da;         // team variation in offensive home adv
  real<lower=0> sigma_dd;         // team variation in defensive home adv
  real mu;
  real<lower=0> sigma_att;
  real<lower=0> sigma_def;
  vector[J] da_raw;      // NCP offensive home advantage
  vector[J] dd_raw;      // NCP defensive home advantage
  vector[J] attack_raw;
  vector[J] defense_raw;
}

transformed parameters {
  vector[J] delta_att = delta_att_mu + sigma_da * da_raw;
  vector[J] delta_def = delta_def_mu + sigma_dd * dd_raw;
  vector[J] attack    = sigma_att * attack_raw;
  vector[J] defense   = sigma_def * defense_raw;
}

model {
  delta_att_mu ~ normal(0.2, 0.5);   // same as iter 9
  delta_def_mu ~ normal(0.0, 0.3);   // weakly centered at 0 (unknown direction)
  sigma_da ~ normal(0, 0.3);
  sigma_dd ~ normal(0, 0.15);        // very tight: home defense advantage small
  mu ~ normal(0.35, 0.5);
  sigma_att ~ normal(0, 0.5);
  sigma_def ~ normal(0, 0.5);
  da_raw      ~ normal(0, 1);
  dd_raw      ~ normal(0, 1);
  attack_raw  ~ normal(0, 1);
  defense_raw ~ normal(0, 1);

  for (n in 1:N_train) {
    real lambda_home = exp(mu + delta_att[home_train[n]] + attack[home_train[n]] - defense[away_train[n]]);
    real lambda_away = exp(mu + attack[away_train[n]] - defense[home_train[n]] - delta_def[home_train[n]]);
    goals_home_train[n] ~ poisson(lambda_home);
    goals_away_train[n] ~ poisson(lambda_away);
  }
}

generated quantities {
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lambda_home = exp(mu + delta_att[home_test[n]] + attack[home_test[n]] - defense[away_test[n]]);
    real lambda_away = exp(mu + attack[away_test[n]] - defense[home_test[n]] - delta_def[home_test[n]]);
    log_lik[n] = poisson_lpmf(goals_home_test[n] | lambda_home);
    log_lik[N_test + n] = poisson_lpmf(goals_away_test[n] | lambda_away);
  }
}
