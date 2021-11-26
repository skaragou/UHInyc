data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> H;
  int<lower=0> K;
  vector[N,H,K] X;
  vector[N,H] y;;
}
parameters {
  real[H] bi[K];
  real sigma;
  // real<lower=0> sigma
  // real<lower=0> eta
  matrix[H,H] Eta;
}
model {
  Eta <- diag_matrix(rep_vector(1.0,K)); 

  for (i in 2:H) {
    b[i,:] ~ multi_normal(b[i-1,:],Eta);
  }

  for (i in 1:N) {
    y[i] ~ normal(b[i,:] * X[i,:],sigma)
  }

} 
