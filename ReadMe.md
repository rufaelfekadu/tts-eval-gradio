# TTS Evaluation Tool

A web-based tool for evaluating text-to-speech (TTS) model outputs through pairwise comparisons or sequential ratings.

## Overview

This tool provides a Gradio-based web interface for conducting listening tests to evaluate the quality of TTS models. It supports two evaluation modes:

1. **Pairwise comparison**: Present two audio samples side by side and have users choose which they prefer.
2. **Sequential rating**: Present audio samples one at a time and have users rate each on a 5-point scale.

## Requirements

- Python 3.7+
- Gradio
- Other standard Python libraries (os, datetime, random, glob, argparse)

Install the required packages:

```bash
pip install gradio
```

## Running the Tool

### Basic Usage

To start the evaluation interface with default settings:

```bash
python main.py
```

This will launch a Gradio interface with the default configuration.

### Command Line Arguments

Customize the tool's behavior using these command line arguments:

- `--out_folder`: Directory to save per-user evaluation results (default: "./evaluation_results")
- `--out_file`: File to save all evaluation results (default: "./annotation.csv")
- `--sample_dirs`: List of directories containing audio samples from different models (default: ["model_a", "model_b"])
- `--completion_code`: Completion code for participants (default: "C103IYNJ")
- `--comparison_mode`: Evaluation method - "pairwise" or "sequential" (default: "sequential")
- `--shuffle`: Whether to shuffle the sample order (default: True)

### Examples

Run with pairwise comparison mode:

```bash
python main.py --comparison_mode pairwise
```

Specify custom model directories:

```bash
python main.py --sample_dirs model1_output model2_output model3_output
```

Full example with multiple options:

```bash
python main.py --comparison_mode sequential \
               --sample_dirs model_a model_b model_c \
               --out_folder results \
               --completion_code ABC123 \
               --shuffle True
```

## File Structure

Your audio sample files should be organized in separate directories for each model/system:

```
model_a/
  ├── sample1.wav
  ├── sample2.wav
  ├── ...
model_b/
  ├── sample1.wav
  ├── sample2.wav
  ├── ...
```

**Important**: All model directories must contain the same number of audio samples with matching names.

## Output Format

Evaluation results are saved in two formats:

1. Individual user results in `[out_folder]/[username]_[speaks_arabic]_[gender]_eval.txt`
2. Combined results in the file specified by `--out_file`

## Integration with Prolific

The tool includes integration with Prolific for crowdsourced evaluations. Participants receive a completion code when they finish the evaluation.
