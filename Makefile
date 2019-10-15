data:
	python make_data.py
	python make_configs.py

run:
	python main.py --config configs/config_nop.ini --verbose

all: clear data

	for f in config_binary.ini config_binomial.ini config_count.ini config_metric.ini config_ordinal.ini; do\
	  python main.py --config configs/$$f --verbose;\
	done

clear:
	rm -rf artificial_data
	rm -rf logs
	rm -rf *experiment_files
	rm -f *.ini
