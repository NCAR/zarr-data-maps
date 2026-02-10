all: build

build:
	PYTHONPATH=${PWD}/python_funcs:${PYTHONPATH} \
	python template_cmip_metrics.py \
	-i /glade/campaign/ral/hap/nlybarger/ESM_eval_postproc/cmip5/ \
	-o output_foo/ \
	-s historical \
	--cmip cmip5

ml:
	@echo "To load modules:"
	@echo "ml conda; conda activate npl"
