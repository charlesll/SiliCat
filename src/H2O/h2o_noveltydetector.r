setwd("/Users/closq/Dropbox/NeuralNetwork/Viscosity/H2O")
library("h2o")
localH2O = h2o.init()

visco_train = h2o.importFile(localH2O, path = "/Users/closq/Dropbox/NeuralNetwork/Viscosity/data/H2O_train.txt")
visco_test = h2o.importFile(localH2O, path = "/Users/closq/Dropbox/NeuralNetwork/Viscosity/data/H2O_test.txt")
visco_valid = h2o.importFile(localH2O, path = "/Users/closq/Dropbox/NeuralNetwork/Viscosity/data/H2O_valid.txt")

ae_visco <- h2o.deeplearning(x = 1:15, 
	training_frame = visco_train, 
	validation_frame = visco_valid, 
	activation = c("Tanh"), 
	hidden = c(12),
	autoencoder = T,
	#input_dropout_ratio = .1,
	#hidden_dropout_ratios = .1,
	epochs = 1000, 
	adaptive_rate = T,
	l1 = 0.0002, 
	#l2 = 0.002, 
	shuffle_training_data = T, 
	reproducible = T,
	loss = "Automatic",
	overwrite_with_best_model = T,
	rho = 0.995,
	epsilon = 1e-8,
	initial_weight_distribution = "UniformAdaptive",
	regression_stop  = 0.000001, 
	diagnostics = T,
	fast_mode = T,
	force_load_balance = T,
	single_node_mode = F,
	quiet_mode = F,
	sparse = F,
	col_major = F,
	average_activation = 0,
	sparsity_beta = 0,
	export_weights_and_biases = TRUE)
		   

train_rec_error <- as.data.frame(h2o.anomaly(novelty,visco_train))
valid_rec_error <- as.data.frame(h2o.anomaly(novelty,visco_valid))
test_rec_error <- as.data.frame(h2o.anomaly(novelty,visco_test))

printerror <- function(data, rec_error, rows) {
  row_idx <- order(rec_error[,1],decreasing=F)[rows]
  my_rec_error <- rec_error[row_idx,]
  my_data <- as.data.frame(data[row_idx,])
  print(my_data)
  print(my_rec_error)
}

printerror(visco_train, train_rec_error, c(4390:4400))
