# Artificial Intelligence Specialization

![banner](./readme/images/readme-banner.jpg)

This repository contains two machine learning projects: a Linear Regression analysis on wave energy farms and a Convolutional Neural Network (CNN) for image classification.

## 📚 Documentation

Detailed analysis for both projects can be found in the `docs` directory:

- Linear Regression: `docs/ml_portfolio.pdf`
- Neural Network: `docs/cnn.pdf`

### 🛠️ Built With

- **Pandas**: data manipulation and analysis
- **Numpy**: numerical computing tols
- **Plotly**: interactive data visualization
- **Matplotlib & Seaborn**: static data visualization
- **Numba**: just-in-time compilation for performance optimization
- **Tensorflow**: machine learning framework

## 📊 Linear Regression: Wave Energy Farm Analysis

### Data Source

- **Dataset**: "Large-scale Wave Energy Farm"
- **Origin**: UCI Machine Learning Repository
- **Contributors**: Researchers from University of Adelaide and Monash University
- **Date Added**: September 16, 2023
- **Instances**: 63,600 unique wave energy converter configurations

### 🚀 Quick Start

#### Prerequisites

- Python 3.11
- pip (Python package manager)

#### Setup

1. Clone the repository:

```bash
git clone https://github.com/salgue441/artificial-intelligence
cd artificial-intelligence
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Run the Analysis

To execute the analysis, run the following command:

```bash
cd regression/src
python main.py
```

## 🖼️ Neural Network: Intel Image Classification

### Data Source

- **Dataset**: Intel Image Classification
- **Origin**: Analytics Vidhya Challenge
- **Instances**: 25,000 images of natural scenes

### 🚀 Quick Start

#### Prerequisites

- Python 3.11
- Cuda 11.2
- cuDNN 8.2

#### Setup

1. Ensure you're in the project root directory:

```bash
cd artificial-intelligence
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Start the Jupyter Notebook server:

```bash
jupyter notebook neural-network/nn.ipynb
```

## 📄 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.
