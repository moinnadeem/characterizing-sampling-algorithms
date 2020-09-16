all: topk nucleus tempered random-topk max-ent random-mask noised-topk target-ent
.PHONY: all
.EXPORT_ALL_VARIABLES:

SHELL:=/bin/bash
.ONESHELL:
include reproduce_envs
	
topk: 
	@echo "=========================================================================================="
	@echo "===================================== TOP-K SAMPLING ====================================="
	@echo "=========================================================================================="

	for BASE in $${K_BASE_VARS[@]} ; do
			export BASE
			python3 main.py FixedSampler 
	done
	unset BASE
	
tempered: 
	@echo "=========================================================================================="
	@echo "================================== TEMPERATURE SAMPLING =================================="
	@echo "=========================================================================================="

	BASE=1.0
	IS_TOP_P=1
	export BASE IS_TOP_P

	for TEMPERATURE in $${TEMPERATURE_VARS[@]} ; do
		export TEMPERATURE 
		python3 main.py TemperatureSweep 
	done
	unset BASE IS_TOP_P TEMPERATURE

nucleus: 
	@echo "=========================================================================================="
	@echo "===================================== TOP-P SAMPLING ====================================="
	@echo "=========================================================================================="

	IS_TOP_P=1
	export IS_TOP_P
	for BASE in $${P_BASE_VARS[@]} ; do
		export BASE 
		python3 main.py FixedSampler
	done
	unset BASE IS_TOP_P MAX_RANDOMNESS

random-topk: 
	@echo "=========================================================================================="
	@echo "============================= RANDOM TOP-K SAMPLING ======================================"
	@echo "=========================================================================================="
	 
	unset IS_TOP_P
	BASE=1
	export BASE
	for MAX_RANDOMNESS in $${K_RANDOMNESS_VARS[@]} ; do
		export BASE MAX_RANDOMNESS 
		echo BASE $$BASE MAX_RANDOMNESS $$MAX_RANDOMNESS RandomFixedSampler 
		python3 main.py RandomFixedSampler
	done
	unset BASE MAX_RANDOMNESS

noised-topk: 
	@echo "==========================================================================================="
	@echo "============================= Sorted Noised SAMPLING ======================================" 
	@echo "==========================================================================================="

	for BASE in 2 3 5 7 10 20 30 50 60 70 80 100 200 300 400 500 600 800 1000 2000 5000 10000 ; do
		for NOISE_WEIGHT in 0.001 0.002 0.003 0.005 ; do
			export BASE NOISE_WEIGHT
			echo BASE $$BASE NOISE_WEIGHT $$NOISE_WEIGHT SortedNoisedFixedSampler 
			python3 main.py SortedNoisedFixedSampler
		done
	done

max-ent: 
	@echo "=========================================================================================="
	@echo "=========================== MAX ENTROPY SAMPLING =========================================" 
	@echo "=========================================================================================="

	for BASE in 0.01 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 2.75 3 3.25 3.5 3.75 4 4.5; do
		export BASE 
		echo BASE $$BASE MaxEntropySampler
			python3 main.py MaxEntropySampler
	done

random-mask: 
	@echo "=========================================================================================="
	@echo "======================== Random Masked Topk SAMPLING ====================================="
	@echo "=========================================================================================="

	PRESERVE_LARGEST_PROB=1
	export PRESERVE_LARGEST_PROB
	echo PRESERVE_LARGEST_PROB $$PRESERVE_LARGEST_PROB
	sleep 2 
	for BASE in 50257 ; do
		for RANDOMSPACE_RATE in 0.001 0.01 0.02 0.05 0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.97 0.99 ; do
			export BASE RANDOMSPACE_RATE
			echo BASE $$BASE SPACE $$RANDOMSPACE_RATE RandomSpaceTopkSampler 
			python3 main.py RandomSpaceTopkSampler
		done
	done

target-ent: 
	@echo "=========================================================================================="
	@echo "=========================== TARGET ENTROPY SAMPLING ======================================"
	@echo "=========================================================================================="

	for BASE in 0.01 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 2.75 3 3.25 3.5 3.75 4 4.5; do
		export BASE 
		echo BASE $$BASE TargetEntropySampler
		python3 main.py TargetEntropySampler
	done
