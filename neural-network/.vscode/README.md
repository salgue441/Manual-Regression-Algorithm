# Artificial Intelligence Specialization

![banner](./readme/images/readme-banner.jpg)

## Projects

### Documentation

The analysis of both projects can be found in the `docs` directory. The `ml_portfolio.pdf` file contains the documentation for the Linear Regression project, while the `cnn.pdf` file contains the documentation for the Neural Network project.

### Linear Regression

#### Data Source

The "Large-scale Wave Energy Farm" dataset, obtained from the UCI Machine Learning Repository, forms the foundation of this analysis. Created by researchers from the University of Adelaide and Monash University, this dataset was contributed to UCI on September 16, 2023. It comprises 63,600 instances, each representing unique configurations of wave energy converters in a wave farm.

#### How to Run

##### Dependencies

- Python 3.11
- Pandas 2.2.0
- Numpy 1.26.3
- Plotly 5.18.0
- Matplotlib 3.8.2
- Seaborn 0.13.1
- Numba 0.58.1

To execute the analysis, run the following command:

```bash
cd regression/src
python main.py
```

or in the codespace configured in this repository.

### Neural Network

The "Intel Image Classification" dataset, obtained from the Intel Image Classification Challenge on Analytics Vidhya, forms the foundation of this analysis. This dataset comprises 25,000 images, each representing a unique scene from a natural environment. The dataset is divided into three categories: training, testing, and validation.

#### How to Run

##### Dependencies

- Python 3.11
- Tensorflow 2.17
- CUDA 11.2
- cuDNN 8.2

##### Execution

To execute the analysis, run the following command:

```bash
cd neural_network
jupyter notebook nn.ipynb
```
