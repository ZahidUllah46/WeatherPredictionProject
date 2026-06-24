# Weather Prediction Using Backpropagation Neural Network

## Project Overview

Weather forecasting plays a vital role in agriculture, transportation, disaster management, and daily planning. This project presents a Weather Prediction System developed using a Neural Network implemented from scratch with the Backpropagation learning algorithm.

The objective of this project is to predict weather conditions based on meteorological features such as temperature, humidity, wind speed, visibility, pressure, and dew point. Instead of relying on high-level deep learning frameworks, the neural network was manually implemented to gain a deeper understanding of machine learning fundamentals and neural network training processes.


##Objectives

* Develop a weather prediction model using Artificial Neural Networks.
* Implement the Backpropagation algorithm from scratch.
* Train and evaluate the model using real-world weather data.
* Understand forward propagation, loss calculation, and weight optimization.
* Analyze model performance using classification metrics.

## System Architecture

### Neural Network Structure

| Layer        | Neurons |
| ------------ | ------- |
| Input Layer  | 6       |
| Hidden Layer | 10      |
| Output Layer | 50      |

### Learning Parameters

* Learning Rate: 0.1
* Epochs: 2000
* Training Algorithm: Backpropagation
* Classification Type: Multi-Class Classification



## Dataset Information

| Attribute        | Value |
| ---------------- | ----- |
| Total Records    | 8,784 |
| Total Features   | 6     |
| Weather Classes  | 50    |
| Training Samples | 7,027 |
| Testing Samples  | 1,757 |

The dataset contains multiple weather conditions including:

* Clear
* Cloudy
* Mainly Clear
* Mostly Cloudy
* Rain
* Snow
* Fog
* Thunderstorms
* And several combined weather conditions


## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Neural Networks
* Backpropagation Algorithm


## Project Workflow

1. Load Weather Dataset
2. Data Cleaning and Preprocessing
3. Feature Selection
4. Label Encoding
5. Data Normalization
6. Train-Test Split
7. Neural Network Initialization
8. Forward Propagation
9. Backpropagation
10. Weight Updates
11. Model Evaluation
12. Weather Prediction


## Results

### Training Performance

* Initial Loss: 4.2561
* Final Loss: 1.8108

### Model Accuracy

* Training Accuracy: ~30%
* Testing Accuracy: 32.27%

The model successfully learned weather patterns and demonstrated continuous improvement during training through loss reduction and weight optimization.


##  Performance Evaluation

The model was evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

The best performance was achieved on major weather categories such as:

* Mainly Clear
* Cloudy
* Mostly Cloudy
* Clear


## How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/WeatherPredictionProject.git
```

### Step 2: Navigate to Project Directory

bash
cd WeatherPredictionProject


### Step 3: Install Dependencies

bash
pip install -r requirements.txt


### Step 4: Run Project

bash
python main.py



## 📁 Project Structure

text
WeatherPredictionProject/
│
├── dataset/
│   └── weather.csv
│
├── main.py
├── neural_network.py
├── preprocessing.py
├── requirements.txt
├── README.md
│
└── results/
    └── output.txt


##  Future Improvements

* Increase prediction accuracy using advanced architectures.
* Apply multiple hidden layers.
* Implement ReLU and Softmax activation functions.
* Handle class imbalance using data augmentation techniques.
* Compare results with modern machine learning algorithms.
* Deploy the model as a web application.

---

## Learning Outcomes

This project provided practical experience in:

* Artificial Intelligence
* Machine Learning
* Neural Networks
* Backpropagation
* Data Preprocessing
* Model Evaluation
* Python Programming

---

## Author

**Zahid Ullah**

Computer Science Student | AI & Machine Learning Enthusiast

Passionate about building intelligent systems, solving real-world problems, and exploring the future of Artificial Intelligence.



## License

This project is developed for educational and learning purposes. Feel free to use, modify, and improve it for academic or research activities.
