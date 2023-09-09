# Google Summer of Code 2023: <br/> Differentiating Real and Misaligned Introns with Machine Learning
### Organisation
Genome Assembly and Annotation (European Bioinformatics Institute / EMBL-EBI)

### Research Group
Ensembl - <a href="https://www.ebi.ac.uk/about/teams/genome-interpretation/">Genome Interpretation Teams</a>

### Mentors
<a href="https://www.ebi.ac.uk/people/person/jose-manuel-gonzalez-martinez/">Jose Gonzalez</a><br/>
<a href="https://www.ebi.ac.uk/people/person/jonathan-mudge/">Jonathan Mudge</a><br/>
<a href="https://www.ebi.ac.uk/people/person/adam-frankish/">Adam Frankish</a>

## Abstract 


## Getting Started


### Installation
- Download the latest zipped script in `Releases`
- Install the required dependencies detailed below

### Requirements - ION
**Please Note that this requirement is for the ION script in `Releases`, not the notebooks**\
<br/>
biopython==1.81\
matplotlib==3.5.2\
numpy==1.23.1\
numpy==1.25.0\
pandarallel==1.6.5\
pandas==1.4.3\
pandas==2.0.3\
pyBigWig==0.3.22\
pyfaidx==0.7.2.1\
scikit_learn==1.1.1\
shap==0.41.0\
tqdm==4.64.0\
xgboost==1.6.2\

### Usage
> bb = pyBigWig.open("https://www.encodeproject.org/files/ENCFF001JBR/@@download/ENCFF001JBR.bigBed")

## Background
Understanding the impact of genetic variation on disease requires comprehensive gene annotation.
Human genes are well characterised following more than two decades of work on their annotation, however,
we know that this annotation is not complete and that new experimental methods are generating data
to help us towards the goal of complete gene annotation.

The advancement in the accuracy of long-read sequencing technology has allowed us to explore novel transcript variants of known genes.
Preventing potentially wrong transcripts and gene annotation is essential to the science community as many rely on the annotation for decision-making.
Automated workflow with a has been developed to minimise the time needed to verify and annotated those transcript variants. However,
current workflows are developed using a very strict rule-set and hence many of the novel transcript variants were rejected.
This project aims to address this issue by using machine learning to differentiate good quality but rejected transcripts,
using it as a standalone classification filter or analysing the decision-making methods of the model and consequently
improving the rule-set used in the automated workflow.

You can read more about the background of this project at: https://github.com/jmgonzmart/GSoC_ML_gene_annot

## Feature Engineering
### Data Preprocessing
### Features Used
### Feature Selection

## Machine Learning Model
### Model Architecture
### Training
### Hyper-parameter Optimisation
### Evaluation Metrics

## Results
### Internal Validation & Benchmark
### External Validation & Benchmark

## References

## Acknowledgements and Notes
