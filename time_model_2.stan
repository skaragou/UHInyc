data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> H;
  int<lower=0> K;
  vector[K] X[N];
  vector[H] y[N];
  vector[M] y_cov[H];
  row_vector[K] beta_mean;
  row_vector[K] beta_sd;
}
transformed data{
  int T = N - M;
  vector[K] X_t[M] = X[1:M,:];
  // vector[K] X_new[M] = X[M + 1:N,:];

  vector[H] y_t[M] = y[1:M,:];
    // vector[H] y_new[T] = y[M + 1:N,:];
}
parameters {
  matrix[H, K] beta_raw;
  real<lower=0> sigma_b[K];
  real<lower=0> sigma_y;
}
transformed parameters{
  matrix[H, K] beta;
  cov_matrix[H] Sigma;
  Sigma = gp_exp_quad_cov(y_cov, 1,1);
  
  row_vector[K] tmp;
  beta[1, ] = beta_mean + beta_sd .* beta_raw[1, ]; 

  for(t in 2:H) {
    tmp = beta[t - 1,];
    beta[t,] = tmp + to_row_vector(sigma_b) .* beta_raw[t,];  
  }
}
model {
  sigma_b ~ normal(0, 1);
  to_vector(beta_raw) ~ normal(0, 1);

  for (i in 1:M) {
    y_t[i,:] ~ multi_normal(beta * X_t[i,:],Sigma);
  }
} 
generated quantities {

  vector[H] y_rep[M];
  for (i in 1:M) {
    y_rep[i,:] = multi_normal_rng(beta * X_t[i,:],Sigma);
  }
}