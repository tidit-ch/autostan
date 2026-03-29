// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 5: Correlated attack/defense via bivariate NCP.
// Strong teams have high attack AND low defense (easy to score against = high defense[j]).
// Bivariate normal with negative correlation corr_ad expected:
//   attack[j] = sigma_att * z_att[j]
//   defense[j] = sigma_def * (corr_ad * z_att[j] + sqrt(1 - corr_ad^2) * z_def[j])
// With only ~11 games/team, correlation borrows info across attack/defense dims.
// Base: iter 3 NCP hierarchical Poisson (NLPD=1.5463, best so far).

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
  real<lower=-1, upper=1> corr_ad;   // attack-defense correlation
  vector[J] z_att;
  vector[J] z_def;
}

transformed parameters {
  // Bivariate NCP: attack and defense share a latent team quality dimension
  vector[J] attack  = sigma_att * z_att;
  vector[J] defense = sigma_def * (corr_ad * z_att + sqrt(1.0 - corr_ad^2) * z_def);
}

model {
  home_adv ~ normal(0, 1);
  mu ~ normal(0, 1);
  sigma_att ~ normal(0, 1);
  sigma_def ~ normal(0, 1);
  corr_ad ~ normal(-0.3, 0.4);  // weakly negative: strong attackers also defend well
  z_att ~ normal(0, 1);
  z_def ~ normal(0, 1);

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
