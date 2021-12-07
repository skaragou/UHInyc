data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> H;
  int<lower=0> K;
  int<lower=0> L;
  int<lower=0> S;
  vector[K] X[N];
  vector[H] y[N];
  vector[K] X_val[M];
  row_vector[K] beta_mean;
  row_vector[K] beta_sd;
}
parameters {
  matrix[H, K] beta_raw;
  real<lower=0> sigma_b[K];
  real<lower=0> sigma_y;
}
transformed parameters{
  matrix[H, K] beta;
  row_vector[K] tmp;
  beta[1, ] = beta_mean + beta_sd .* beta_raw[1, ]; 

  for(t in 2:H) {
    tmp = beta[t - 1,];
    beta[t,] = tmp + beta_raw[t,];  
  }
}
model {
  sigma_y ~ inv_gamma(1,1);
  to_vector(beta_raw) ~ normal(0, 1);

  for (i in 1:M) {
    y[i,:] ~ normal(beta * X[i,:],sigma_y);
  }
} 
generated quantities {
  real y_rep[L,S,H];
  real y_hat[M,H];
  real log_p = 0;
  
  for (i in 1:M) {
    log_p += normal_lpdf(y[i,:] | beta * X[i,:],sigma_y);
  }

  for (l in 1:L) {
    for (i in 1:S) {
      y_rep[l,i,:] = normal_rng(beta * X[i,:],sigma_y);
    }
  }

  for (i in 1:M) {
    y_hat[i,:] = normal_rng(beta * X_val[i,:],sigma_y);
  }
}