Additional Material for the publication:

# An Evaluation of State-of-the-Art Approaches to Relation Extraction for Usage on Domain-Specific Corpora
## Christoph Brandl, [Jens Albrecht](https://www.th-nuernberg.de/en/person/albrecht-jens/) and Renato Budinich

This publication was created as part of the research group [Future Engineering](https://www.th-nuernberg.de/einrichtungen-gesamt/fraunhofer-forschungsgruppen/future-engineering/).

### Manually Labelled Future Engineering Data & Adapted FewRel Data

The folder 'fe-training-data' contains all available examples from our manually labelled Future Engineering data. They are splitted into training, test and evaluation data files. The data set is based on articles extracted from [electrive.com](https://www.electrive.com/faq-electrive), a news provider targeting decision-makers, manufacturers and service providers in the e-mobility sector.

In addition, the folder 'fewrel-training-data' contains the used training and evaluation data from the [FewRel](https://www.zhuhao.me/fewrel/) data set, as described in the conference papers.

### Implementations of Different Relation Extraction Approaches

This repository contains different approaches for the Relation Extraction task from text.
At the moment the repository contains working implementations of the following approaches :

* [Entity-aware BLSTM](https://arxiv.org/abs/1901.08163) based on [this GitHub repository](https://github.com/roomylee/entity-aware-relation-classification)
* [ERNIE](https://arxiv.org/abs/1905.07129) based on [this GitHub repository](https://github.com/thunlp/ERNIE)
* [R-BERT](https://arxiv.org/abs/1905.08284) based on [this GitHub repository](https://github.com/mickeystroller/R-BERT)
* [Matching the Blanks BERT](https://arxiv.org/abs/1906.03158) based on the [this GitHub repository](https://github.com/plkmo/BERT-Relation-Extraction)
* [BERT Pair](https://arxiv.org/abs/1910.07124) based on [this GitHub repository](https://github.com/thunlp/FewRel)

In addition the repository contains a converter for parsing TSV files from the [INCEptTION](https://inception-project.github.io/) annotation tool transfering them into a data format similar to the format of [FewRel](https://www.zhuhao.me/fewrel/) data.

&nbsp;

### Requirements

---

* python == 3.6
* torch >= 1.5.0
* transformers == 3.0.0
* nltk >= 3.2.5
* rdflib >= 5.0.0
* tagme >= 0.1.3
* flair >= 0.6.0
* wptools >= 0.4.17
* pydotplus >= 2.0.2
* graphviz >= 0.10.1
* lime >= 0.2.0.1

There is a requiremets.txt file included in the repository for installing all needed libraries in the correct version.  
However, note that some of the libraries can not be installed via a requirements file and have to be installed seperately. In particular, **PyTorch**, **Flair** and **PyCurl**.

&nbsp;

### Installation

---

In order to use the approaches in this repository some additional files like pretraining checkpoints or additional data sources of the approaches have to be downloaded.

The Matching the Blanks GitHub repository provides a data file for the pre-training process of the BERT model:

* [Pre-training data for MTB training](https://drive.google.com/file/d/1aMiIZXLpO7JF-z_Zte3uH7OCo4Uk_0do/view?usp=sharing)

The authors of the ERNIE approach provide additional data:

* [Pre-trained knowledge embeddings](https://drive.google.com/open?id=14VNvGMtYWxuqT-PWDa8sD0e7hO486i8Y)
* [Pre-trained ERNIE model](https://drive.google.com/open?id=1DVGADbyEgjjpsUlmQaqN6i043SZvHPu5)

The used data for fine-tuning the approaches to the specific tasks are also provided:

* [Training data for FewRel task](https://drive.google.com/file/d/1PsUJJejGVGak--h7bglO34U0ol_7w5fU/view?usp=sharing)
* [Training data for Future Engineering task](https://drive.google.com/file/d/1rfZPmzdhMSeS6DlLPcEUtc20KCQMDnyL/view?usp=sharing)

The Entity-aware BLSTM approach uses pre-trained Glove vectors for word representation (the extracted file should be located in a resource folder inside the approaches folder):

* [GloVe pre-trained word vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip)

&nbsp;

The dowloaded data can be extracted and moved into the corresponding folder of the approach in the repository.

&nbsp;

### Usage

---

Each of the above approaches is included in an own Jupyter notebook. There the approach can be trained on one of the datasets (fine-tuning). At the end of those notebooks all needed information including the trained model weights and additional resources is stored in checkpoint files. This  training step is a prerequisite for using the models later for the inference of new sentences in the Text2RelationGraph notebook.

The notebook Text2RelationGraph contains a complete processing from a not annotated text to RDF-Triples building a knowledge graph. Therefore one of the approaches can be chosen dynamically within the notebook. The notebook uses the previously trained and stored information from the approaches individual notebooks.  
Additionally an evaluation of all approaches can be done with different datasets. Metrics as accuracy, precision, recall and F1 score are calculated and a confusion matrix is plotted.
