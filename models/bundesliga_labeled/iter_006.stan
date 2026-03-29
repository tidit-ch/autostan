// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 6: NCP + tighter half-normal(0, 0.5) on sigma_att/sigma_def.
// Iter 1 (centered + sigma~N+(0,0.5)) got 1.5465.
// Iter 3 (NCP + sigma~N+(0,1)) got 1.5463 (best).
// Combining NCP geometry with tighter regularization on sigma.
// Also: informative priors on mu and home_adv from observed training means.

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
}

transformed parameters {
  vector[J] attack  = sigma_att * attack_raw;
  vector[J] defense = sigma_def * defense_raw;
}

model {
  // Weakly informative: training data shows mu~log(1.42)=0.35, home_adv~0.21
  home_adv ~ normal(0.2, 0.5);
  mu ~ normal(0.35, 0.5);
  // Tighter sigma: combines NCP (iter3) with half-normal(0,0.5) (iter1)
  sigma_att ~ normal(0, 0.5);
  sigma_def ~ normal(0, 0.5);
  attack_raw  ~ normal(0, 1);
  defense_raw ~ normal(0, 1);

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
