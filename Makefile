
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

TF_MODEL_FOLDER = TF_Model/
MODEL_FILENAME = TF_Model/tf_model.meta
CONV_SCRIPT = ./convert-mnist.py
RUN_SCRIPT = ./predict-mnist-ncsdk2.py
TRAIN_SCRIPT = ./train-minst.py
INPUT_NODE_FLAG = -in=conv2d_1_input
OUTPUT_NODE_FLAG = -on=dense_2/Softmax
KERAS_WEIGHT_FILE = weights.h5
KERAS_MODEL_JSON_FILE = model.json

.PHONY: all
all: profile check compile

.PHONY: prereqs
prereqs:
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' ${RUN_SCRIPT}
	@chmod +x ${RUN_SCRIPT}

.PHONY: profile
profile: weights
	${NCPROFILE} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: browse_profile
browse_profile: weights profile
	@if [ -e output_report.html ] ; \
	then \
		firefox output_report.html & \
	else \
		@echo "***\nError - output_report.html not found" ; \
	fi ; 

.PHONY: weights
weights: train
	@sed -i 's/\r//' ${CONV_SCRIPT}
	@chmod +x ${CONV_SCRIPT}
	(test -f ${KERAS_WEIGHT_FILE} && test -f ${KERAS_MODEL_JSON_FILE}) || (echo "Please run \'make train\' first.")
	test -f ${MODEL_FILENAME} || ${CONV_SCRIPT}

.PHONY: compile
compile: weights
	test -f graph || ${NCCOMPILE} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: check
check: weights
	${NCCHECK} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: run
run: compile
	python3 ${RUN_SCRIPT}

.PHONY: run_py
run_py: compile
	python3 ${RUN_SCRIPT}

.PHONY: train
train:
	(test -f ${KERAS_WEIGHT_FILE} && test -f ${KERAS_MODEL_JSON_FILE}) || python3 ${TRAIN_SCRIPT}
	@echo "Training finished. \
	If you want to retrain the model, delete '${KERAS_WEIGHT_FILE}' and '${KERAS_MODEL_JSON_FILE}' files."

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile, check, cpp, run_py, run_cpp";
	@echo "  make weights - downloads the trained model";
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser.";
	@echo "  make run_py - runs the run.py python example program";
	@echo "  make clean - removes all created content"


clean: 
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f *.ckpt
	rm -f output_result.npy
	rm -f output_val.csv
	rm -rf output
	rm -f graph
	rm -rf ${TF_MODEL_FOLDER}
	rm -f ${KERAS_WEIGHT_FILE}
	rm -f ${KERAS_MODEL_JSON_FILE}
  
