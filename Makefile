data:
	python -m HyBayes --make_data --verbose
	python -m HyBayes --make_configs --verbose

all: clear data

	for f in config_binary.ini config_binomial.ini config_count.ini config_metric.ini config_ordinal.ini; do\
	  time python  -m HyBayes --config configs/$$f --verbose;\
	done

clear:
	rm -rf artificial_data
	rm -rf logs
	rm -rf *experiment_files
	rm -f *.ini
