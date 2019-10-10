data:
	python make_data.py
run:
	python main.py --config configs/config_nop.ini --verbose
all: clear data
	for f in configs/*; do\
	  python main.py --config $$f ;\
	done
clear:
	rm -rf artificial_data
	rm -rf logs
	rm -rf *experiment_files
	rm -f *.log
	rm -f *.ini
