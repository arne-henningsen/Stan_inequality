// data
data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x1;
  vector[N] x2;
  real lt; // lower threshold
  real rs; // regularisation sigma
}

// define parameters
parameters {
  real b0;
  real b1;
  real b2;
  real b11;
  real b12;
  real b22;
  real<lower=0> sigma;
}

// partial derivatives
transformed parameters{
  vector[N] dydx1;
  vector[N] dydx2;
  for( i in 1:N ) {
    dydx1[i] = b1 + b11 * x1[i] + b12 * x2[i];
    dydx2[i] = b2 + b12 * x1[i] + b22 * x2[i];
  }
}

// specify the model
model {
  // priors
  b0 ~ normal( 0, 5 );
  b1 ~ normal( 0.5, 5 );
  b2 ~ normal( 0.5, 5 );
  b11 ~ normal( 0, 1 );
  b12 ~ normal( 0, 1 );
  b22 ~ normal( 0, 1 );
  sigma ~ normal( 0, 10 );
  // soft constraints on the partial derivatives
  for( i in 1:N ) {
    if( dydx1[i] < lt ) {
        target += normal_lpdf( dydx1[i] - lt | 0, rs );
    } else {
        // Adding a constant to ensure smoothness around `lower_threshold`
        target += normal_lpdf( 0 | 0, rs );     
    }
    if( dydx2[i] < lt ) {
        target += normal_lpdf( dydx2[i] - lt | 0, rs );
    } else {
        // Adding a constant to ensure smoothness around `lower_threshold`
        target += normal_lpdf( 0 | 0, rs );     
    }
  }  

  // fitted values
  vector[N] mu;
  for( i in 1:N ) {
    mu[i] = b0 + b1 * x1[i] + b2 * x2[i] + 0.5 * b11 * x1[i]^2 
      + b12 * x1[i] * x2[i] + 0.5 * b22 * x2[i]^2;
  }
  y ~ normal( mu, sigma );
}
