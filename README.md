# Meta-Learning for Better Learning: Using Meta-Learning Methods to Automatically Label Exam Questions with Detailed Learning Objectives

This repository contains the code and material for the Meta-Learning for Better Learning: 
Using Meta-Learning Methods to Automatically Label Exam Questions with Detailed Learning Objectives paper. 

## Dataset and Benchmark

The files for the collected learning objective + course question dataset can be found in the following directory names: 
`OpenStax Dataset`, `Principles of Chemistry 3rd edition`, and `Chem 31A`.

Functions for loading the respective files are provided in the `util.py` file.

To access the benchmark datasets, see the `openstax_dataset.py` file.

## Training a ProtoTransformer

Use the `trainer.py` to train a classifier on the 2-way k-shot classification task of labeling course questions with learning objectives.

To see parameters for the training script, run
`python trainer.py -h`

A template training run is provided below:
`python trainer.py --log_dir <OUTPUT DIRECTORY> --model_size <MODEL SIZE (e.g. tiny, bert)> --num_support <K> --num_query <1-10> --batch_size <BATCH SIZE> --num_workers <1-8> --num_epochs <1-10> --learning_rate <~1e-5>`

### Testing on Held-Out Test Set

The `trainer.py` file provides a test option as follows:
`python trainer.py --log_dir <OUTPUT_DIRECTORY> ... --test`

One can use the `--split` flag to choose between the `test`, `train`, and `val` datasets.

### Testing on Held-Out Course

Provide a course name, along with the `--test` flag, to test the classifier on a particular course:
`python trainer.py --log_dir <OUTPUT_DIRECTORY> ... --test --course_name <COURSE_NAME (e.g. Chem 31A)>`

## Testing a GPT-3 Classifier

Coming soon!