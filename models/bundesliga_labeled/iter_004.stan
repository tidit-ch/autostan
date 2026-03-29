// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 4: Dixon-Coles correction for low-score matches.
// Observed (0,0)=15 vs ~8.7 expected, strong excess of goalless draws.
// Dixon-Coles (1997) multiplies the joint density by tau(x,y) for
// (x,y) in {0,1}x{0,1}.
// Base: NCP hierarchical Poisson (iter 3, NLPD=1.5463).

functions {
  // Log of Dixon-Coles tau correction for match result (x, y)
  real dc_tau_log(int x, int y, real lam_h, real lam_a, real rho) {
    if (x == 0 && y == 0) return log(1.0 - lam_h * lam_a * rho);
    if (x == 1 && y == 0) return log(1.0 + lam_a * rho);
    if (x == 0 && y == 1) return log(1.0 + lam_h * rho);
    if (x == 1 && y == 1) return log(1.0 - rho);
    return 0.0;
  }
}

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
  real<lower=0, upper=0.3> rho;  // DC parameter; bounded so tau(0,0)>0 for typical lambda
}

transformed parameters {
  vector[J] attack  = sigma_att * attack_raw;
  vector[J] defense = sigma_def * defense_raw;
}

model {
  home_adv ~ normal(0, 1);
  mu ~ normal(0, 1);
  sigma_att ~ normal(0, 1);
  sigma_def ~ normal(0, 1);
  attack_raw  ~ normal(0, 1);
  defense_raw ~ normal(0, 1);
  rho ~ beta(2, 10);  // prior on [0, 0.3], centers ~0.17

  for (n in 1:N_train) {
    real lambda_home = exp(mu + home_adv + attack[home_train[n]] - defense[away_train[n]]);
    real lambda_away = exp(mu + attack[away_train[n]] - defense[home_train[n]]);
    target += dc_tau_log(goals_home_train[n], goals_away_train[n], lambda_home, lambda_away, rho);
    target += poisson_lpmf(goals_home_train[n] | lambda_home);
    target += poisson_lpmf(goals_away_train[n] | lambda_away);
  }
}

generated quantities {
  // log_lik[n]          = home contribution + full DC correction
  // log_lik[N_test + n] = away contribution (no extra correction)
  // Sum per match = joint log-likelihood under DC model
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lambda_home = exp(mu + home_adv + attack[home_test[n]] - defense[away_test[n]]);
    real lambda_away = exp(mu + attack[away_test[n]] - defense[home_test[n]]);
    log_lik[n] = poisson_lpmf(goals_home_test[n] | lambda_home)
                 + dc_tau_log(goals_home_test[n], goals_away_test[n], lambda_home, lambda_away, rho);
    log_lik[N_test + n] = poisson_lpmf(goals_away_test[n] | lambda_away);
  }
}
