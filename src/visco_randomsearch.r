setwd("/Users/closq/Dropbox/NeuralNetwork/Viscosity/H2O")
library("h2o")
localH2O = h2o.init()

visco_train = h2o.importFile(localH2O, path = "/Users/closq/Dropbox/NeuralNetwork/Viscosity/data/H2O_train.txt")
visco_valid = h2o.importFile(localH2O, path = "/Users/closq/Dropbox/NeuralNetwork/Viscosity/data/H2O_valid.txt")

models <- c()
saverho <- c()
for (i in 1:100) {
  rand_activation <- c("Rectifier")#c("Tanh","Rectifier")[sample(1:2,1)]
  rand_numlayers <- 3#sample(2,1)
  rand_hidden <- c(sample(2:15,rand_numlayers,T))
  rand_l1 <- runif(1, 1e-5, 1e-2)
  #rand_l2 <- runif(1, 1e-5, 1e-2)
  rand_rho <- runif(1,0.980,0.999)
  saverho <- c(saverho,rand_rho)
  dlmodel <- h2o.deeplearning(
  				x = 1:15, y = 16, 	
				training_frame = visco_train, 
				validation_frame = visco_valid, 
  				epochs=1000,
                                activation=rand_activation, 
				hidden=rand_hidden, 
				l1=rand_l1, #l2=rand_l2,
				adaptive_rate = T,
				shuffle_training_data = T, 
				reproducible = T,
				loss = "Automatic",
				initial_weight_distribution = "UniformAdaptive",
				overwrite_with_best_model = T,
				rho = rand_rho,
                              )
  models <- c(models, dlmodel)
}

best_err <- 100
for (i in 1:length(models)) {
  err <- h2o.mse(models[[i]],valid=TRUE)
  if (err < best_err) {
      best_err <- err
      best_model <- models[[i]]
      best_rho <- saverho[[i]]
    }
}