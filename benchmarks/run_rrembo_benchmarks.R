# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""
Runs benchmarks for REMBO variance implemented in RRembo (k_\Psi kernel, with
standard projection and with \gamma back projection). 

RRembo should be installed from https://github.com/mbinois/RRembo .

Code below is based on examples/Tests_RRembo.R from that repo.
"""

library("doParallel")
library("foreach")
library("RRembo")
library("DiceKriging")
library("rjson")


######################################## Branin 100

nCores <- 7
cl <-  makeCluster(nCores)
registerDoParallel(cl)

nrep <- 50
popsize <- 80
gen <- 40
roll <- TRUE

d <- 2
D <- 100
budget <- 50
covtype <- "matern5_2"
ftest <- branin_mod

mat_effective <- c(20, 65)
fstar <- 0.397887

lower <- rep(0, D)
upper <- rep(1, D)

###

cat('REMBO standard k_Psi \n')
# Standard REMBO-like method
tsP <- Sys.time()
res_standard_kPsi <- foreach(i=1:nrep, .packages = c("RRembo", "DiceKriging"), .combine = 'rbind') %dopar% {
  set.seed(i)
  res <- easyREMBO(
    par = runif(d),
    fn = ftest,
    budget = budget,
    lower = lower,
    upper = upper,
    ii = mat_effective,
    control = list(
      Atype = 'standard',
      reverse = FALSE,
      warping = 'Psi',
      testU = FALSE,
      standard = TRUE,
      popsize = popsize,
      gen = gen,
      inneroptim = "StoSOO",
      roll = roll
    ),
    kmcontrol = list(covtype = covtype)
  )
  res$y
}
tsP <- difftime(Sys.time(), tsP, units = "sec")
print(paste0('-------------------------', tsP))

###

cat('REMBO reverse + A Gaussian + Psi \n')
trP <- Sys.time()
res_reverse_kPsi <- foreach(i=1:nrep, .packages = c("RRembo", "DiceKriging"), .combine = 'rbind') %dopar% {
  set.seed(i)
  res <- easyREMBO(
    par = runif(d),
    fn = ftest,
    budget = budget,
    lower = lower,
    upper = upper,
    ii = mat_effective,
    control = list(
      Atype = 'Gaussian',
      reverse = TRUE,
      warping = 'Psi',
      testU = TRUE,
      standard = FALSE,
      popsize = popsize,
      gen = gen,
      inneroptim = "StoSOO",
      roll = roll
    ),
    kmcontrol = list(covtype = covtype)
  )
  res$y
}
trP <- difftime(Sys.time(), trP, units = "sec")
print(paste0('-------------------------', trP))

###

stopCluster(cl)

res_sP <- list()
for (i in 1:50) {
  res_sP[[i]] <- res_standard_kPsi[i,]
}

res_rP <- list()
for (i in 1:50) {
  res_rP[[i]] <- res_reverse_kPsi[i,]
}

write(toJSON(res_sP), file="results/branin_100_rrembos_standard_kPsi.json")
write(toJSON(res_rP), file="results/branin_100_rrembos_reverse_kPsi.json")


######################################## Hartmann6 100

nCores <- 7
cl <-  makeCluster(nCores)
registerDoParallel(cl)

nrep <- 50
popsize <- 80
gen <- 40
roll <- TRUE

d <- 6
D <- 100
budget <- 200
covtype <- "matern5_2"
ftest <- hartman6_mod

mat_effective <- c(20, 15, 44, 38, 67, 4)
fstar <- -3.32237

lower <- rep(0, D)
upper <- rep(1, D)
  
###

cat('REMBO standard k_Psi \n')
# Standard REMBO-like method
tsP <- Sys.time()
res_standard_kPsi <- foreach(i=1:nrep, .packages = c("RRembo", "DiceKriging"), .combine = 'rbind') %dopar% {
  set.seed(i)
  res <- easyREMBO(
    par = runif(d),
    fn = ftest,
    budget = budget,
    lower = lower,
    upper = upper,
    ii = mat_effective,
    control = list(
      Atype = 'standard',
      reverse = FALSE,
      warping = 'Psi',
      testU = FALSE,
      standard = TRUE,
      popsize = popsize,
      gen = gen,
      inneroptim = "StoSOO",
      roll = roll
    ),
    kmcontrol = list(covtype = covtype)
  )
  res$y
}
tsP <- difftime(Sys.time(), tsP, units = "sec")
print(paste0('-------------------------', tsP))


###

cat('REMBO reverse + A Gaussian + Psi \n')
trP <- Sys.time()
res_reverse_kPsi <- foreach(i=1:nrep, .packages = c("RRembo", "DiceKriging"), .combine = 'rbind') %dopar% {
  set.seed(i)
  res <- easyREMBO(
    par = runif(d),
    fn = ftest,
    budget = budget,
    lower = lower,
    upper = upper,
    ii = mat_effective,
    control = list(
      Atype = 'Gaussian',
      reverse = TRUE,
      warping = 'Psi',
      testU = TRUE,
      standard = FALSE,
      popsize = popsize,
      gen = gen,
      inneroptim = "StoSOO",
      roll = roll
    ),
    kmcontrol = list(covtype = covtype)
  )
  res$y
}
trP <- difftime(Sys.time(), trP, units = "sec")
print(paste0('-------------------------', trP))

###

stopCluster(cl)

res_sP <- list()
for (i in 1:50) {
  res_sP[[i]] <- res_standard_kPsi[i,]
}

res_rP <- list()
for (i in 1:50) {
  res_rP[[i]] <- res_reverse_kPsi[i,]
}

write(toJSON(res_sP), file="results/hartmann6_100_rrembos_standard_kPsi.json")
write(toJSON(res_rP), file="results/hartmann6_100_rrembos_reverse_kPsi.json")


######################################## Hartmann6 1000

nCores <- 7
cl <-  makeCluster(nCores)
registerDoParallel(cl)

nrep <- 50
popsize <- 80
gen <- 40
roll <- TRUE

d <- 6
D <- 1000
budget <- 200
covtype <- "matern5_2"
ftest <- hartman6_mod

mat_effective <- c(191, 141, 431, 371, 661, 31)
fstar <- -3.32237

lower <- rep(0, D)
upper <- rep(1, D)
  
###

cat('REMBO standard k_Psi \n')
# Standard REMBO-like method
tsP <- Sys.time()
res_standard_kPsi <- foreach(i=1:nrep, .packages = c("RRembo", "DiceKriging"), .combine = 'rbind') %dopar% {
  set.seed(i)
  res <- easyREMBO(
    par = runif(d),
    fn = ftest,
    budget = budget,
    lower = lower,
    upper = upper,
    ii = mat_effective,
    control = list(
      Atype = 'standard',
      reverse = FALSE,
      warping = 'Psi',
      testU = FALSE,
      standard = TRUE,
      popsize = popsize,
      gen = gen,
      inneroptim = "StoSOO",
      roll = roll
    ),
    kmcontrol = list(covtype = covtype)
  )
  res$y
}
tsP <- difftime(Sys.time(), tsP, units = "sec")
print(paste0('-------------------------', tsP))


### Too slow, not run

# cat('REMBO reverse + A Gaussian + Psi \n')
# trP <- Sys.time()
# res_reverse_kPsi <- foreach(i=1:nrep, .packages = c("RRembo", "DiceKriging"), .combine = 'rbind') %dopar% {
#   set.seed(i)
#   res <- easyREMBO(
#     par = runif(d),
#     fn = ftest,
#     budget = budget,
#     lower = lower,
#     upper = upper,
#     ii = mat_effective,
#     control = list(
#       Atype = 'Gaussian',
#       reverse = TRUE,
#       warping = 'Psi',
#       testU = TRUE,
#       standard = FALSE,
#       popsize = popsize,
#       gen = gen,
#       inneroptim = "StoSOO",
#       roll = roll
#     ),
#     kmcontrol = list(covtype = covtype)
#   )
#   res$y
# }
# trP <- difftime(Sys.time(), trP, units = "sec")
# print(paste0('-------------------------', trP))

###

stopCluster(cl)

res_sP <- list()
for (i in 1:50) {
  res_sP[[i]] <- res_standard_kPsi[i,]
}

write(toJSON(res_sP), file="results/hartmann6_1000_rrembos_standard_kPsi.json")
