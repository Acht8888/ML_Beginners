# ML_Beginners

## GitHub Repository

[Repository Link](https://github.com/Acht8888/ML_Beginners.git)

## Team Members and Task Distribution

| Name                 | Student ID | Tasks Assigned                                                      | Contributions |
| -------------------- | ---------- | ------------------------------------------------------------------- | ------------- |
| Nguyễn Quang Duy     | 2252120    | Model implementation (Neural Networks), documentation, final review | 20%           |
| Văn Duy Anh          | 2252045    | Model implementation (Naive Bayes)                                  | 20%           |
| Nguyễn Đoàn Hải Băng | 2252078    | Model implementation (Genetic Algorithms)                           | 20%           |
| Trần Vũ Hảo          | 2052978    | Model implementation (Decision Trees)                               | 20%           |
| Đào Tiến Tuấn        | 1953069    | Model implementation (Graphical Models)                             | 20%           |

## Project Overview

### Models Implemented

- Decision Trees
- Neural Networks
- Naive Bayes
- Genetic Algorithms
- Graphical Models (Bayesian Networks, HMM)

### Key Objectives

- Hands-on experience in implementing multiple ML models.
- Compare performance and interpretability of different models.
- Focus on engineering practices, model implementation, and comparative analysis.

## Environment Setup

### Step 1: Create a Virtual Environment

```bash
python -m venv env
```

### Step 2: Activate the Virtual Environment

On Windows (Command Prompt)

```bash
env\Scripts\activate
```

On macOS/Linux

```bash
source env/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train a Model

You can train a machine learning model using the `train` command. It requires the following arguments:

#### Example:

```bash
python src/main.py train --model_type neural_network --model_name neural_network_1 --mode manual --lr 0.001 --batch_size 32 --epochs 20 --hidden_size 15
```

#### Arguments:

- `--model_type`: The type of model to train. Choose from:

  - `decision_tree`
  - `neural_network`
  - `naive_bayes`
  - `genetic_algorithm`
  - `graphical_model`

- `--model_name`: The name to give to the trained model.

- `--mode`: Select between `manual` and `study` for how to load hyperparameters:

  - `manual`: Manually input hyperparameters for training.
  - `study`: Load hyperparameters from a study file.

- `--lr`: Learning rate for training. (Only in `manual` mode)

- `--batch_size`: Batch size for training. (Only in `manual` mode)

- `--hidden_size`: Hidden layer size for the model (Only in `manual` mode).

- `--epochs`: Number of epochs to train. (Only in `manual` mode).

- `--file_name`: File name to load the hyperparameters if using `study` mode.

#### Example with study mode:

```bash
python src/main.py train --model_type neural_network --model_name neural_network_2 --mode study --file_name study_data.json
```

---

### 2. Tune Hyperparameters

Use the `tune` command to perform hyperparameter tuning on an already trained model.

#### Example:

```bash
python src/main.py tune --model_type neural_network --model_name neural_network_2 --trials 20 --direction maximize
```

#### Arguments:

- `--model_type`: The type of model to tune. Choose from:

  - `decision_tree`
  - `neural_network`
  - `naive_bayes`
  - `genetic_algorithm`
  - `graphical_model`

- `--model_name`: The name of the trained model to tune.

- `--trials`: The number of tuning trials (e.g., how many different hyperparameter configurations to try).

- `--direction`: Whether to minimize or maximize the tuning objective:
  - `minimize`: To minimize loss.
  - `maximize`: To maximize accuracy.

---

### 3. Visualize Model Performance

You can visualize the performance of the model using the `visualize` command. This command can visualize either the tuning process or the evaluation of a trained model.

#### Example:

```bash
python src/main.py visualize --mode tune --file_name tuning_results.json
```

#### Arguments:

- `--mode`: Choose between:

  - `tune`: Visualize the results from hyperparameter tuning.
  - `evaluate`: Visualize the evaluation of the trained model.

- `--file_name`: The file name containing model data (either tuning results or evaluation results).

---

### 4. Evaluate a Trained Model

The `evaluate` command allows you to evaluate the performance of a trained model.

#### Example:

```bash
python src/main.py evaluate --file_name trained_model.json
```

#### Arguments:

- `--file_name`: The file name of the trained model to evaluate.

---

## Additional Commands

You can also run a prediction on a given dataset:

```bash
python src/main.py predict --model neural_network --input_data data/sample.csv
```

This will use the trained model to make predictions on the input dataset.

---

## CLI Command Examples:

- **Train a neural network manually**:

  ```bash
  python src/main.py train --model_type neural_network --model_name neural_network_1 --mode manual --lr 0.001 --batch_size 32 --epochs 20 --hidden_size 15
  ```

- **Train a model using hyperparameters from a study file**:

  ```bash
  python src/main.py train --model_type neural_network --model_name neural_network_2 --mode study --file_name study_data.json
  ```

- **Tune hyperparameters for a trained neural network model**:

  ```bash
  python src/main.py tune --model_type neural_network --model_name neural_network_2 --trials 20 --direction maximize
  ```

- **Visualize the results of the tuning process**:

  ```bash
  python src/main.py visualize --mode tune --file_name tuning_results.json
  ```

- **Evaluate a trained model**:
  ```bash
  python src/main.py evaluate --file_name trained_model.json
  ```

---
