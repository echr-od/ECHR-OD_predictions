# Repository structure

```
root
   |
   |----- data
   |        |----- analysis                            <-- folder containing the analysis of raw output
   |        |        |----- cm                         <-- confusion matrix plots
   |        |        |----- lc                         <-- learning curves plots
   |        |        |----- tables                     <-- latex tables
   |        |----- input                               <-- folder containing the datasets from ECHR-OD
   |        |----- output                              <-- folder containing the experiments raw output
   |        |        |----- result_binary.json         <-- results for binary datasets
   |        |        |----- result_multiclass.json     <-- results for multiclass dataset
   |        |        |----- result_multilabel.json     <-- results for multilabel dataset
   |----- echr_experiments
   |        |----- config.py                           <-- configuration for experiments and analysis
   |        |----- data.py                             <-- helpers to load datasets
   |        |----- format.py                           <-- helpers to format raw output
   |        |----- plot.py                             <-- helpers to plot results
   |        |----- scorers.py                          <-- information to keep from each experiement
   |        |----- utils.py                            <-- helpers to update raw output and save analysis files
   |----- binary_experiments.py                        <-- script to run experiments on binary datasets
   |----- binary_confusion_matrices.py                 <-- script to plot confusion matrices from binary output
   |----- binary_generate_latex.py                     <-- script to generate latex tables from binary output
   |----- binary_learning_curves.py                    <-- generate the learning curves for the best methods on each binary dataset
   |----- binary_statistics.py                         <-- generate statistics about the binary datasets
   |----- multiclass_experiments.py                    <-- script to run experiments on multiclass dataset
   |----- multiclass_confusion_matrices.py             <-- script to plot confusion matrices from multiclass output
   |----- multiclass_generate_latex.py                 <-- script to generate latex tables from multiclass output
   |----- multilabel_experiments.py                    <-- script to run experiments on multilabel dataset
   |----- multilabel_generate_latex.py                 <-- script to generate latex tables from multilabel output
   |----- multilabel_plot_results.py                   <-- script to plot results from multilabel output
   |----- README.md
   |----- .gitignore
```


# Usage and replication with Docker

Build the image using:
```
docker build -f Dockerfile -t echr_experiments .
```

To run the experiments:
```
docker run -it --rm --mount src=$(pwd),dst=/tmp/echr_experiments/,type=bind --mount src=<ECHR_OD_BUILD>,dst=/tmp/echr_experiments/data/input/,type=bind echr_experiments run <task>
```
where `<ECHR_OD_BUILD>` is the **absolute** path to ECHR-OD build to use and `<task>` is the classification task to solve among `binary`, `multiclass` and `multilabel`.


To run the experiments:
```
docker run -it --rm --mount src=$(pwd),dst=/tmp/echr_experiments/,type=bind --mount src=<ECHR_OD_BUILD>,dst=/tmp/echr_experiments/data/input/,type=bind echr_experiments analyze <task>
```
where `<ECHR_OD_BUILD>` and `<task>` are as define above.


# Configuration

The experimental environment variables are defined in ```echr_experiments/config.py```. If you wish to replicate  the experiments from the article *European Court of Human Rights Open Data project*, leave the default value.

If you want to reproduce all experiments and analysis from scratch, remove the files in ```/data/output/``` and in ```/data/analysis/```:
```
rm data/output/*
rm data/analysis/cm/*
rm data/analysis/lc/*
rm data/analysis/tables/*
```

Download all datasets from the [ECHR-OD project](https://echr-opendata.eu) and place them in ```/data/input/```.

Run the script for the three classification problems:
```
python binary_experiments.py
python multiclass_experiments.py
python multilabel_experiments.py
```
The json file in ```/data/output``` are updated for every couple (dataset flavor, algorithm) such that if the experiment crashes, there is no need to restart it from scratch. Also, it allows to analyze partial results before the end of the experiements.

**REMARK**: It took several days to complete all the experiments on a laptop equiped with a i7-6820HQ @ 2,70GHz and 32GB RAM. The experiments on binary and multilabel datasets are run in parallel such that the CPU is used as much as possible. This is not the case for multiclass as for the analysis we collect the confusion matrix and there is a problem with concurrent access. Therefore, multiclass experiments are much slower.

Run the script to analyze the results and generate the plots and LaTeX tables:
```
python binary_generate_latex.py
python binary_confusion_matrices.py
python binary_learning_curves.py
python multiclass_generate_latex.py
python multiclass_confusion_matrices.py
python multilabel_generate_latex.py
```

## Selecting your methods

If you would like to run the experiments on a subset of methods, open the configuration file ```echr_experiments/config.py```) and search for the definition of the variable ```<X>_CLASSIFIERS``` where ```X``` stands for the problem version (```BINARY```, ```MULTICLASS``` or ```MULTILABEL```). You can then comment or uncomment the methods, as well as configuring them rather than leaving the default parameters.

