data {
  int<lower=0> M;
  int<lower=0> T;
  int<lower=0> K;
  int<lower=0> S;
  int<lower=0> L;
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
  real log_p = normal_lpdf(y | X * beta, sigma);
  real y_out[T] = normal_rng(X_val * beta, sigma);
  real y_rep[S,L];

  for (i in 1:S) {
    y_rep[i,] = normal_rng(X[:L,] * beta, sigma);
  }
}