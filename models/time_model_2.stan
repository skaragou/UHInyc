data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> H;
  int<lower=0> K;
  int<lower=0> L;
  vector[K] X[N];
  vector[H] y[N];
  row_vector[K] beta_mean;
  // row_vector[K] zeros;
  row_vector[K] beta_sd;
}
transformed data{
  int T = N - M;
  vector[K] X_t[M] = X[1:M,:];
  // vector[K] X_new[M] = X[N-M:N,:];

  vector[H] y_t[M] = y[1:M,:];
  // vector[H] y_new[M] = y[N-M:N,:];
}
parameters {
  matrix[H, K] beta_raw;
  real<lower=0> sigma_b[K];
  real<lower=0> sigma_y;
  // cov_matrix[H] sigma_y;
}
transformed parameters{
  matrix[H, K] beta;
  // matrix[H, K] nu;
  row_vector[K] tmp;
  beta[1, ] = beta_mean + beta_sd .* beta_raw[1, ]; 

  for(t in 2:H) {
    tmp = beta[t - 1,];
    beta[t,] = tmp + to_row_vector(sigma_b) .* beta_raw[t,];  
  }

  // nu[1, ] = beta_mean + beta_sd .* beta_raw[1, ]; 
  // beta[1, ] = zeros; 

  // for(t in 2:H) {
  //   tmp = nu[t - 1,];
  //   nu[t,] = tmp + to_row_vector(sigma_b) .* beta_raw[t,];  
  // }

  // for(t in 2:H) {
  //   beta[t,] = beta[t-1,] + nu[t-1, ];  
  // }

}
model {
  sigma_b ~ normal(0,1);
  sigma_y ~ inv_gamma(1,1);
  to_vector(beta_raw) ~ normal(0, 1);

  for (i in 1:M) {
    y_t[i,:] ~ normal(beta * X_t[i,:],sigma_y);
  }
} 
generated quantities {

  // vector[H] y_rep[M];
  real y_rep[L,M,H];

  for (l in 1:L) {
    for (i in 1:M) {
      y_rep[l,i,:] = normal_rng(beta * X_t[i,:],sigma_y);
    }
  }
  // real y_hat[M,H];
  // for (i in 1:M) {
  //   y_hat[i,:] = normal_rng(beta * X_new[i,:],sigma_y);
  // }
}