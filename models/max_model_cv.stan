data {
  int<lower=0> T;
  int<lower=0> M;
  int<lower=0> K;
  real sigma_y;
  matrix[M,K] X;
  vector[M] y;
  matrix[T,K] X_val;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0,1);
  sigma ~ inv_gamma(1,1);
  y ~ normal(X * beta, sigma); 
} 
generated quantities {
  real y_out[T] = normal_rng(X_val * beta, sigma);
}