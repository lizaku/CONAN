# CONAN

* `Make_tables.ipynb` - jupyter notebook which produces contingency tables for HS and CN types in the dataset.
* `process_dataset_json.py` - takes json file with the dataset and parses it into the train and test sets.
* `tfidf_SVM.py` - script for performing experiments with tf-idf and SVM classifiers. Contains 3 functions - for testing tf-idf, embeddings with cosine similarity ranking and SVM classifier/
* `lstm_siamese` - folder containing all the materials for the NN experiments. `sent_sim.py` - building the model, `test_sent_sim.py` - testing the model.
* `data_sample.py` - script preparing the data for the NN experiments.

The models created during the experiments with the NN classifier are in the folder `lstm_siamese/checkpoints`.
