// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 10: Symmetric home advantage — delta[h] affects both
// home team's attack AND defense.
// lambda_home = exp(mu + delta[h] + attack[h] - defense[a])  [iter 9]
// lambda_away = exp(mu + attack[a] - defense[h] - delta[h])  [NEW: home team defends better]
// Crowd support on defense: home team concedes fewer goals too.
// Same delta[h] for both, so home advantage is a single latent strength.
// Base: iter 9 model (best NLPD=1.5432).

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
  real delta_mu;              // global home advantage
  real<lower=0> sigma_delta;
  real mu;
  real<lower=0> sigma_att;
  real<lower=0> sigma_def;
  vector[J] delta_raw;
  vector[J] attack_raw;
  vector[J] defense_raw;
}

transformed parameters {
  vector[J] delta   = delta_mu + sigma_delta * delta_raw;
  vector[J] attack  = sigma_att * attack_raw;
  vector[J] defense = sigma_def * defense_raw;
}

model {
  delta_mu ~ normal(0.1, 0.3);  // smaller center: now delta affects both sides
  sigma_delta ~ normal(0, 0.3);
  mu ~ normal(0.35, 0.5);
  sigma_att ~ normal(0, 0.5);
  sigma_def ~ normal(0, 0.5);
  delta_raw   ~ normal(0, 1);
  attack_raw  ~ normal(0, 1);
  defense_raw ~ normal(0, 1);

  for (n in 1:N_train) {
    real lambda_home = exp(mu + delta[home_train[n]] + attack[home_train[n]] - defense[away_train[n]]);
    real lambda_away = exp(mu + attack[away_train[n]] - defense[home_train[n]] - delta[home_train[n]]);
    goals_home_train[n] ~ poisson(lambda_home);
    goals_away_train[n] ~ poisson(lambda_away);
  }
}

generated quantities {
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lambda_home = exp(mu + delta[home_test[n]] + attack[home_test[n]] - defense[away_test[n]]);
    real lambda_away = exp(mu + attack[away_test[n]] - defense[home_test[n]] - delta[home_test[n]]);
    log_lik[n] = poisson_lpmf(goals_home_test[n] | lambda_home);
    log_lik[N_test + n] = poisson_lpmf(goals_away_test[n] | lambda_away);
  }
}
