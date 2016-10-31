In this project we try to create a neuronal network to recognize single characters. We use the IAM handwriting database.

Requirements
============
You will need python2, with numpy, matplotlib, sklearn, theano and keras installed. Is advisable to have an acceptable GPU with cudNN activated if available. Example configurations files for theanorc and keras are provided.

Database creation
============
1. Download the raw data from http://www.fki.inf.unibe.ch/databases/iam-handwriting-database, extract it inside rawdata directory.
2. Use hwc_create.py to create the dataset used for the network.
3. Use imgToH5.py to create the h5 files containing the preprocessed datasets.
4. Once all is over the networks can be trained (files with prefix red_)
