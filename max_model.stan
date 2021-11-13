functions {
  array[] int permutation_rng(int N) {
     array[N] int y;
     for (n in 1:N) {
       y[n] = n;
     }
     vector[N] theta = rep_vector(1.0 / N, N);
     for (n in 1:N) {
      int i = categorical_rng(theta);
      array[n] int temp = y;
      y[n] = y[i];
      y[i] = temp[n];
     }
     return y;
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_test;    // number of data items
  vector[N] num_build500;      // outcome vector
  vector[N] mean_fa_ratio;      // outcome vector
  vector[N] min_distance_park;      // outcome vector
  vector[N] num_trees_50m;      // outcome vector
  vector[N] y;      // outcome vector
}
transformed data{
  int N_train = N - N_test;
  int permutation[N] = permutation_rng(N);
  vector[N_test] num_build500_val = num_build500[permutation[N_train + 1 : N]];      // outcome vector
  vector[N_test] mean_fa_ratio_val = mean_fa_ratio[permutation[N_train + 1 : N]];      // outcome vector
  vector[N_test] min_distance_park_val = min_distance_park[permutation[N_train + 1 : N]];      // outcome vector
  vector[N_test] num_trees_50m_val = num_trees_50m[permutation[N_train + 1 : N]];      // outcome vector
  vector[N_test] y_val = y[permutation[N_train + 1 : N]]; 

  vector[N_train] num_build500_train = num_build500[permutation[1 : N_train]];      // outcome vector
  vector[N_train] mean_fa_ratio_train = mean_fa_ratio[permutation[1 : N_train]];      // outcome vector
  vector[N_train] min_distance_park_train = min_distance_park[permutation[1 : N_train]];      // outcome vector
  vector[N_train] num_trees_50m_train = num_trees_50m[permutation[1 : N_train]];      // outcome vector
  vector[N_train] y_train = y[permutation[1 : N_train]]; 
}
parameters {
  real a;
  real b;
  real c;
  real d;
  real bias;
  real<lower=0> sigma;  // error scale
}
model {
  // a ~ beta(1,1);
  // b ~ beta(1,1);
  // c ~ beta(1,1);
  // d ~ beta(1,1);
  // sigma ~ inv_gamma(1,1);
  y_train ~ normal(a * num_build500_train + b * mean_fa_ratio_train + c * min_distance_park_train + d * num_trees_50m_train + bias, sigma); 
}
// generated quantities {
//   real log_p = normal_lpdf(y_val | a * num_build500_val + b * mean_fa_ratio_val + c * min_distance_park_val + d * num_trees_50m_val + bias, sigma);
// }