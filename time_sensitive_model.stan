data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> H;
  int<lower=0> K;
  vector[K] X[N];
  vector[H] y[N];
}
transformed data{
  int T = N - M;
  vector[K] X_t[M] = X[1:M,:];
  // vector[K] X_new[M] = X[M + 1:N,:];

  vector[H] y_t[M] = y[1:M,:];
  // vector[H] y_new[T] = y[M + 1:N,:];
}
parameters {
  cov_matrix[H] Sigma;
  cov_matrix[H] Eta;
  real<lower=0> rho;
}
transformed parameters{
  matrix[H,K] b;
  for (i in 2:H) 
    b[i,:] = multi_normal(b[i-1,:],Eta);
}
model {
  b[1,:] ~ normal(0,rho);

  for (i in 1:M) {
    y_t[i,:] ~ multi_normal(b * X_t[i,:],Sigma);
  }
} 
generated quantities {
  vector[H] y_rep[M];
  for (i in 1:M) {
    y_rep[i,:] = multi_normal_rng(b * X_t[i,:],Sigma);
  }
}