#!/usr/bin/env python

gendata: 
	-rm -r ./data/viscosity/saved_subsets/*.pkl
	-rm -r ./data/viscosity/saved_subsets/*.npy
	python data_generation.py viscosity

.PHONY : clean	
clean:
	-rm -r ./data/viscosity/saved_subsets/*.pkl
	-rm -r ./data/viscosity/saved_subsets/*.npy
	
	-rm -r ./silicat/saved/*.pkl
	-rm -r ./silicat/saved/*.npy
	-rm -r ./silicat/saved/*.h5
	-rm -r ./silicat/saved/*.json
	
cleandata:
	-rm -r ./data/viscosity/saved_subsets/*.pkl
	-rm -r ./data/viscosity/saved_subsets/*.npy

cleanmodel:	
	-rm -r ./silicat/saved/*.pkl
	-rm -r ./silicat/saved/*.npy
	-rm -r ./silicat/saved/*.h5
	-rm -r ./silicat/saved/*.json
	
