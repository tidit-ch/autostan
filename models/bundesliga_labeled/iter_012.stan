// AutoStan: Edit this file to improve NLPD.
// See program.md for instructions.
//
// Iteration 12: ZIP (Zero-Inflated Poisson) for both home and away goals.
// Home zeros: 51 observed vs ~36 expected under Poisson(1.754) — excess of 15.
// Away zeros: 54 observed vs ~50 expected under Poisson(1.420) — slight excess.
// Global pi_zero_home and pi_zero_away parameters for structural zeros.
// MOM estimate: pi_zero_home ≈ 0.088, pi_zero_away ≈ 0.025.
// Base: iter 9 (team-specific home advantage, best NLPD=1.5432).

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
  real delta_mu;
  real<lower=0> sigma_delta;
  real mu;
  real<lower=0> sigma_att;
  real<lower=0> sigma_def;
  vector[J] delta_raw;
  vector[J] attack_raw;
  vector[J] defense_raw;
  real<lower=0, upper=1> pi_h;  // zero-inflation for home goals (MOM ~0.09)
  real<lower=0, upper=1> pi_a;  // zero-inflation for away goals (MOM ~0.02)
}

transformed parameters {
  vector[J] delta   = delta_mu + sigma_delta * delta_raw;
  vector[J] attack  = sigma_att * attack_raw;
  vector[J] defense = sigma_def * defense_raw;
}

model {
  delta_mu ~ normal(0.2, 0.5);
  sigma_delta ~ normal(0, 0.3);
  mu ~ normal(0.35, 0.5);
  sigma_att ~ normal(0, 0.5);
  sigma_def ~ normal(0, 0.5);
  delta_raw   ~ normal(0, 1);
  attack_raw  ~ normal(0, 1);
  defense_raw ~ normal(0, 1);
  pi_h ~ beta(2, 20);  // prior mean ~0.09
  pi_a ~ beta(1, 40);  // prior mean ~0.024

  for (n in 1:N_train) {
    real lam_h = exp(mu + delta[home_train[n]] + attack[home_train[n]] - defense[away_train[n]]);
    real lam_a = exp(mu + attack[away_train[n]] - defense[home_train[n]]);
    // ZIP log-likelihood
    if (goals_home_train[n] == 0) {
      target += log_sum_exp(log(pi_h), log1m(pi_h) + poisson_lpmf(0 | lam_h));
    } else {
      target += log1m(pi_h) + poisson_lpmf(goals_home_train[n] | lam_h);
    }
    if (goals_away_train[n] == 0) {
      target += log_sum_exp(log(pi_a), log1m(pi_a) + poisson_lpmf(0 | lam_a));
    } else {
      target += log1m(pi_a) + poisson_lpmf(goals_away_train[n] | lam_a);
    }
  }
}

generated quantities {
  vector[2 * N_test] log_lik;

  for (n in 1:N_test) {
    real lam_h = exp(mu + delta[home_test[n]] + attack[home_test[n]] - defense[away_test[n]]);
    real lam_a = exp(mu + attack[away_test[n]] - defense[home_test[n]]);
    if (goals_home_test[n] == 0) {
      log_lik[n] = log_sum_exp(log(pi_h), log1m(pi_h) + poisson_lpmf(0 | lam_h));
    } else {
      log_lik[n] = log1m(pi_h) + poisson_lpmf(goals_home_test[n] | lam_h);
    }
    if (goals_away_test[n] == 0) {
      log_lik[N_test + n] = log_sum_exp(log(pi_a), log1m(pi_a) + poisson_lpmf(0 | lam_a));
    } else {
      log_lik[N_test + n] = log1m(pi_a) + poisson_lpmf(goals_away_test[n] | lam_a);
    }
  }
}
