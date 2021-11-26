data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] num_build500;      
  vector[N] mean_fa_ratio;
  vector[N] min_distance_park;
  vector[N] num_trees_50m;
  // vector[N] hour;
  vector[N] y;
}
parameters {
  real a;
  real b;
  real c;
  real d;
  // real f;
  real bias;
  real<lower=0> sigma;  // error scale
}
model {
  // a ~ beta(1,1);
  // b ~ beta(1,1);
  // c ~ beta(1,1);
  // d ~ beta(1,1);
  sigma ~ inv_gamma(1,1);
  y[1:M] ~ normal(a * num_build500[1:M] + b * mean_fa_ratio[1:M] + c * min_distance_park[1:M] + d * num_trees_50m[1:M] +  bias, sigma); 
} 
generated quantities {
  real log_p = normal_lpdf(y[M + 1:N] | a * num_build500[M + 1:N] + b * mean_fa_ratio[M + 1:N] + c * min_distance_park[M + 1:N] + d * num_trees_50m[M + 1:N] + bias, sigma);
  // array[N - M] real y_rep = normal_rng(a * num_build500[M + 1:N] + b * mean_fa_ratio[M + 1:N] + c * min_distance_park[M + 1:N] + d * num_trees_50m[M + 1:N] + f * hour[M + 1:N]+ bias, sigma);
  array[M] real y_rep = normal_rng(a * num_build500[1:M] + b * mean_fa_ratio[1:M] + c * min_distance_park[1:M] + d * num_trees_50m[1:M] + bias, sigma);
}