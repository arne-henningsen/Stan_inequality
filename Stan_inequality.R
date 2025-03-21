# load required package(s)
library( "miscTools" )
library( "rstan" )

# number of observations (before removing unsuitable observations)
nObs <- 500
# explanatory variables
set.seed( 123 )
dat <- data.frame(
  x1 = rnorm( nObs ),
  x2 = rnorm( nObs )
)
# parameters
b0 <- 1
b1 <- 0.5
b2 <- 0.7
b11 <- -0.1
b12 <- 0.2
b22 <- -0.25
sigma <- 0.5
# dependent variable
dat$y <- with( dat, b0 + b1 * x1 + b2 * x2 + 0.5 * b11 * x1^2
  + b12 * x1 * x2 + 0.5 * b22 * x2^2 ) + rnorm( nObs, 0, sigma )
# partial derivatives
dat$dydx1 <- with( dat, b1 + b11 * x1 + b12 * x2 )
dat$dydx2 <- with( dat, b2 + b12 * x1 + b22 * x2 )
par( mfrow = c( 2, 1 ) )
hist( dat$dydx1, 20 )
hist( dat$dydx2, 20 )
par( mfrow = c( 1, 1 ) )
hist( dat$dydx1 + dat$dydx2, 20 )
# remove observations with negative partial derivatives
dat <- subset( dat, dydx1 >= 0 & dydx2 >= 0 )

# prepare the data for Stan
dat_stan <- list(
  N = nrow( dat ),
  y = dat$y,
  x1 = dat$x1,
  x2 = dat$x2
)

# set number of cores
options( mc.cores = parallel::detectCores() )

# compile the model
model_stan <- stan_model( "Stan_inequality.stan" )

# generate initial values
set.seed( 12345 )
init_stan <- list()
for( chain in 1:4 ){
  init_stan[[ chain ]] <- list(
    b1 = runif( 1, min = 0.2, max = 1 ),
    b2 = runif( 1, min = 0.2, max = 1 ),
    b11 = runif( 1, min = -0.02, max = 0.02 ),
    b12 = runif( 1, min = -0.02, max = 0.02 ),
    b22 = runif( 1, min = -0.02, max = 0.02 )
  )
}

# fit the model
fit_stan <- sampling( model_stan, data = dat_stan, iter = 2000, chains = 4,
  init = init_stan, control = list( adapt_delta = 0.999, max_treedepth = 20 ) )
mean( get_divergent_iterations( fit_stan ) )
print( fit_stan, pars = grep( "^[^d]", fit_stan@model_pars, value = TRUE ) )
plot( fit_stan, plotfun = "rhat" )
plot( fit_stan, pars = grep( "^[^d]", fit_stan@model_pars, value = TRUE ),
  plotfun = "trace", inc_warmup = TRUE )
suppressWarnings(
  pairs( fit_stan, pars = grep( "^[^d]", fit_stan@model_pars, value = TRUE ) ) )
  
# investigate first derivatives (across all observations)
samples_stan <- as.data.frame( fit_stan )
par( mfrow = c( 2, 1 ) )
hist( unlist( samples_stan[ , grep( "^dydx1", names( samples_stan ) ) ] ), 30 )
hist( unlist( samples_stan[ , grep( "^dydx2", names( samples_stan ) ) ] ), 30 )

# compare estimated partial derivatives with 'true' partial derivatives
summary_stan <- summary( fit_stan )$summary
# mean estimates
dat$dydx1estMean <- summary_stan[ grep( "^dydx1", rownames( summary_stan ) ), "mean" ]
dat$dydx2estMean <- summary_stan[ grep( "^dydx2", rownames( summary_stan ) ), "mean" ]
all.equal( dat$dydx1estMean, with( dat, summary_stan[ "b1", "mean" ] +
    summary_stan[ "b11", "mean" ] * x1 + summary_stan[ "b12", "mean" ] * x2 ) )
all.equal( dat$dydx2estMean, with( dat, summary_stan[ "b2", "mean" ] +
    summary_stan[ "b12", "mean" ] * x1 + summary_stan[ "b22", "mean" ] * x2 ) )
summary( dat$dydx1estMean )
summary( dat$dydx2estMean )
compPlot( dat$dydx1, dat$dydx1estMean )
compPlot( dat$dydx2, dat$dydx2estMean )
# median estimates
dat$dydx1estMedian <- summary_stan[ grep( "^dydx1", rownames( summary_stan ) ), "50%" ]
dat$dydx2estMedian <- summary_stan[ grep( "^dydx2", rownames( summary_stan ) ), "50%" ]
summary( dat$dydx1estMedian )
summary( dat$dydx2estMedian )
compPlot( dat$dydx1, dat$dydx1estMedian )
compPlot( dat$dydx2, dat$dydx2estMedian )
