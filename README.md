<div align="center">
<img src="static/r2r_logo_updated.png" width="250" alt="R2R Logo" />
<h1>Ensuring Medical AI Safety: Explainable AI-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data</h1>

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) &nbsp;&nbsp; [![PyTorch](https://img.shields.io/badge/PyTorch-1.13-brightgreen)](https://pytorch.org/)
</div>

## Description

Deep neural networks are increasingly employed in high-stakes medical applications, despite their tendency for shortcut learning in the presence of spurious correlations, which can have potentially fatal consequences.
Detecting and mitigating shortcut behavior is a challenging task that often requires significant labeling efforts from domain experts.
To alleviate this problem, we introduce a semi-automated framework for the identification of spurious behavior from both data and model perspective by leveraging insights from eXplainable Artificial Intelligence (XAI).
This allows the retrieval of spurious data points and pinpointing the model circuits that encode the associated prediction rules. 
Moreover, we demonstrate how these shortcut encodings can be used for XAI-based sample- and pixel-level data annotation, providing valuable information for bias mitigation methods to unlearn the undesired shortcut behavior.
We show the applicability of our framework using four medical datasets across two modalities, featuring controlled and real-world spurious correlations caused by data artifacts.
We successfully identify and mitigate these biases in VGG16, ResNet50, and contemporary Vision Transformer models, ultimately increasing their robustness and applicability for real-world medical tasks.

<div align="center">
    <img src="static/title_figure_v4.png" style="max-width: 1000px; width: 100%;" alt="R2R Overview" />
    <p>Overview of our bias annotation framework extending the Reveal2Revise framework.</p>
</div>

## Table of Contents

- [Description](#description)
- [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Datasets](#datasets)
- [Model Training (optional)](#model-training-optional)
- [Preprocessing](#preprocessing)
- [Concept Validation (step 1)](#concept-validation-step-1)
- [Bias Modeling and Biased Sample Retrieval (steps 2-4)](#bias-modeling-and-biased-sample-retrieval-steps-2-4)
- [Bias Localization (step 5)](#bias-localization-step-5)
- [Bias Mitigation](#bias-mitigation)

## Prerequisites
### Installation

We use Python 3.10.9 and PyTorch 1.13. To install the required packages, run:

```bash 
pip install -r requirements.txt
```

### Datasets
Secondly, the datasets need to be downloaded. This includes ISIC2019, CheXpert, HyperKvasir, and PTB-XL. 
For the former, download and extract the [ISIC 2019](https://challenge.isic-archive.com/landing/2019/) dataset using the following:

```bash
mkdir datasets
cd datasets
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.zip
unzip ISIC_2019_Training_Input.zip
unzip ISIC_2019_Training_GroundTruth.zip
```

To download [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert), as well as annotations for the `pacemaker` artifact, run the following via Kaggle CLI:

```bash
kaggle datasets download ashery/chexpert
```

We further use labels for the existence of `pacemakers`provided by [this work](https://github.com/nina-weng/FastDiME_Med).

For downloading the [HyperKvasir](https://datasets.simula.no/hyper-kvasir/) dataset, run the folloing:
```bash
wget https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip
unzip hyper-kvasir-labeled-images.zip
```

And lastly, the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset can be downloaded as follows:
```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

For PTB-XL, we followed the preprocessing described [here](https://github.com/hhi-aml/xai4ecg).

## Model Training (optional)

With the required packages installed and the datasets downloaded, the models can be trained. To consolidate training parameters in a unified file, we utilize configuration files (`.yaml`-files). These files specify training hyperparameters, such as architecture, optimizer and learning rate, as well as dataset parameters and output directories. 

We provide scripts to conveniently generate the config files, that can be run as follows (here for ISIC2019):

```bash 
python config_files/training/config_generator_training_isic.py
```

Using the previously generated config files, we can train the models by running:

```bash
python -m model_training.start_training --config_file "config_files/training/isic/your_config_file.yaml"
```

**NOTE**: Instead of training models from scratch, pre-trained models can be downloaded [here](https://datacloud.hhi.fraunhofer.de/s/YgWe6eXS5XELgdK). Provided CAVs and detected outlier neurons are valid for our pre-traiend models only.


## Preprocessing

All bias identification, annotation and mitigation approaches require latent activations or relevance scores, commonly aggregated via max- or avg-pooling. Therefore, we provide a pre-processing script that pre-computes activations and relevances for all considered layers of the networks for the entire dataset. These pre-computed values can be used for example to compute CAVs or to run SpRAy. The script can be run for a given config-file as follows:

 ```bash
python -m experiments.preprocessing.run_preprocessing --config_file "config_files/your_config_file.yaml"
```

## Concept Validation (step 1)
<div align="center">
    <img src="static/data_vs_concept_perspective.png" style="max-width: 600px; width: 100%;" alt="Data vs. Concept Perspective" />
    <p>Data- vs. concept perspective for concept validation</p>
</div>

Detecting biased representations in large models can be challenging, especially without prior knowledge of spurious correlations. A common approach is to identify outlier behavior using a reference dataset.
Automated methods typically analyze either post-hoc explanations of reference images to find anomalies or identify outlier representations within the model itself. In this step, we differentiate between two perspectives:

- **Data Perspective:** Focuses on detecting samples that exhibit outlier behavior, e.g., *samples* of dermoscopic images containing band-aids.
- **Model Perspective:** Aims to identify outlier concept representations within the model, e.g., *neurons* extracting the band-aid concept.

It's important to note that outlier behavior may not always indicate spurious correlations; it can represent valid but infrequently used model behavior. Therefore, human experts often need to manually inspect detected outliers to determine their validity.

In our experiments, we consider SpRAy both in input and latent space for the data perspective. For the model perspective, we consider the analysis of activation pattern, i.e., finding neurons activating upon similar signal, via DORA, and the analysis of relevance pattern by grouping neurons by their relevance scores for predictions. 

The scripts can be run as follows:

```bash
CONFIG="config_files/revealing/isic/your_config_file.yaml"
# 1) Data perspective: Run SpRAy
# Note: the config file defines the layer, specifying whether SpRAy is run in input or latent space
python -m experiments.preprocessing.run_spray_preprocessing --config_file $CONFIG --class_indices "0,1"
python -m experiments.reveal.spray.plot_spray_outliers_2d --config_file $CONFIG --class_id 0

# 2) Model perspective (activation pattern): Run DORA
python -m experiments.preprocessing.run_dora_preprocessing --config_file $CONFIG
python -m experiments.reveal.dora.plot_dora_outliers_2d --config_file $CONFIG --class_id 0

# 3) Model perspective (relevance pattern): Cosine similarity between relevances
python -m experiments.preprocessing.run_crp_preprocessing --config_file $CONFIG
python -m experiments.reveal.crp.plot_crp_outliers_2d --config_file $CONFIG --class_id 0
```

All scripts are accomodated with interactive versions, as further documented in [experiments/reveal](experiments/reveal/README.md). An example for itentified model and dataset weaknesses in ISIC2019 using a ResNet50 model is shown below. Identified neurons encoding spurious concepts are listed in [data/spurious_neurons.json](data/spurious_neurons.json).

<div align="center">
    <img src="static/bias_identification_experiments.png" style="max-width: 1000px; width: 100%;" alt="Bias identification experiments" />
    <p>Model/data weaknesses in ISIC2019 with ResNet50</p>
</div>

## Bias Modeling and Biased Sample Retrieval (steps 2-4)
The bias modeling step is an iterative approach itself and concept representations can be improved by correcting labeling errors in the data.
In this process, samples with high bias scores are manually inspected to enhance label quality. The steps include:
1. Starting with a small set of bias samples identified through concept validation methods (see above).
2. Fitting an initial Concept Activation Vector (CAV).
3. Manually inspecting samples with high bias scores to improve labels.
4. Using the updated labels to iteratively refine the CAV
5. This results in a set of annotated bias samples.

Refer to the figure below for a visual representation of this process.

<div align="center">
    <img src="static/iterative_bias_annotation.png" style="max-width: 1000px; width: 100%;" alt="Bias identification experiments" />
    <p>Iterative bias modeling and sample retrieval.</p>
</div>

Scripts to run the iterative process in an interactive fashion are provided and documented in [experiments/sample_retrieval](experiments/sample_retrieval/README.md). To run the quantitative experiments reporting AUROC and AP scores (see Sec. 6.3 in the paper), run the following script:

```bash
python -m experiments.sample_retrieval.run_biased_sample_ranking --config_file "config_files/revealing/isic/your_config_file.yaml" --artifact "ruler"
```

Pre-trained CAVs for all pre-trained models/datasets are provided [here](https://datacloud.hhi.fraunhofer.de/s/BKKDrfEeEoKFmJ2).
Annotations for detected spurious samples are provided in [data/artifact_samples](data/artifact_samples).

## Bias Localization (step 5)

XAI insights can automate the spatial localization of biased concepts in artifact samples. We represent bias concepts using CAVs and leverage a local attribution method, specifically Layer-wise Relevance Propagation (LRP) to identify these biases in input space.

Our quantitative experiments in controlled settings, i.e., scenarios where ground truth masks for the biases are available, measuring the artifact relevance and intersection over union (IoU), can be run as follows:

```bash
python -m experiments.localization.eval_localization_quantitatively --config_file "config_files/revealing/hyper_kvasir_attacked/your_config_file.yaml" --artifact "artificial"
```

For non-controlled real-world artifacts, such as the `ruler` in ISIC2019, we cannot compute these metrics due to the absence of ground truth data. However, the computed spatial localizations are extremely valuable annotations for subsequent steps of the Reveal2Revise framework, specifically for bias mitigation or (re-)evaluation, and for other applications. They can be computed automatically as follows:

```bash
python -m experiments.localization.localize_artifacts --config_file "config_files/revealing/isic/your_config_file.yaml" --artifact "ruler"
```

Extracted localizations (as heatmap and binary masks) are provided for real-world artifacts in ISIC2019 (`band-aid`, `ruler`, `skin marker`), CheXpert (`pacemaker`) and HyperKvasir (`insertion tube`) in [data/localizations](data/localizations).

## Bias Mitigation
Lastly, we further provide implementations for the bias mitigation step of the Reveal2Revise framework, utilizing annotations generated via our approaches.
We mitigate biases via Right for the Right Reasons (RRR), Right-Reason ClArC (RR-ClArC) and the training free approaches Projective ClArC (P-ClArC) and reactive P-ClArC (rP-ClArC).

Again, we use config-files (`.yaml`) to specify hyperparameters, such as model details, bias mitigation approach, and mitigation parameters, such as &lambda;-values.
These files are located in in `config_files/bias_mitigation_controlled/` and can be generated as follows:

```bash
python -m config_files.bias_mitigation_controlled.config_generator_mitigation_hyper_kvasir_attacked"
```

Having generated the config-files, the bias mitigation step can be performed as follows:

```bash
python -m experiments.mitigation_experiments.start_model_correction --config_file "config_files/bias_mitigation_controlled/hyper_kvasir_attacked/your_config_file.yaml"
```

Lastly, having mitigated the biases, we (re-)evaluate the models by (1) measuring its performance on test sets with different distributions wrt. the biases, i.e., sets with (attacked) and without (clean) the bias in all samples. Moreover, we (2) measure the relevance put onto the artifact region computed via LRP, (3) compute the TCAV score wrt. the bias concept and (4) compute heatmaps for both the original and the corrected model for biased samples. These scripts can be run as follows:

```bash
CONFIG_FILE="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/your_config_file.yaml"
# 1) Evaluate on different subsets (train/val/test) in different settings (clean/attacked)
python -m experiments.evaluation.evaluate_by_subset_attacked --config_file $CONFIG_FILE

# 2) Measure relevance on artifact region in input space
python -m experiments.evaluation.compute_artifact_relevance --config_file $CONFIG_FILE

# 3) Measure TCAV score wrt bias concept
python -m experiments.evaluation.measure_tcav --config_file $CONFIG_FILE

# 4) Compute heatmaps for biased samples for original and corrected models
python -m experiments.evaluation.qualitative.plot_heatmaps --config_file $CONFIG_FILE
```