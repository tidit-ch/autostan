// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 8: Single quality parameter per team (Bradley-Terry / Elo style).
// log(lambda_home) = mu + home_adv + q[h] - q[a]
// log(lambda_away) = mu - q[h] + q[a]  =  mu + q[a] - q[h]
// Strong team: high goals scored, few conceded.
// Half the team parameters vs attack+defense model (18 vs 36).
// With ~11 games/team, stronger regularization may improve generalization.
// Base priors from iter 6 (best NLPD=1.5460).

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
  real<lower=0> sigma_q;
  vector[J] q_raw;   // NCP quality
}

transformed parameters {
  vector[J] q = sigma_q * q_raw;
}

model {
  home_adv ~ normal(0.2, 0.5);
  mu ~ normal(0.35, 0.5);
  sigma_q ~ normal(0, 0.5);
  q_raw ~ normal(0, 1);

  for (n in 1:N_train) {
    real lambda_home = exp(mu + home_adv + q[home_train[n]] - q[away_train[n]]);
    real lambda_away = exp(mu + q[away_train[n]] - q[home_train[n]]);
    goals_home_train[n] ~ poisson(lambda_home);
    goals_away_train[n] ~ poisson(lambda_away);
  }
}

generated quantities {
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lambda_home = exp(mu + home_adv + q[home_test[n]] - q[away_test[n]]);
    real lambda_away = exp(mu + q[away_test[n]] - q[home_test[n]]);
    log_lik[n] = poisson_lpmf(goals_home_test[n] | lambda_home);
    log_lik[N_test + n] = poisson_lpmf(goals_away_test[n] | lambda_away);
  }
}
