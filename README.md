# 🎓 Student Exam Performance Predictor

A machine learning web application that predicts students' math scores based on various demographic and academic factors using Flask and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline that predicts students' mathematics exam scores based on:

- **Gender**
- **Race/Ethnicity** (Group A-E)
- **Parental Level of Education**
- **Lunch Type** (Standard/Free or Reduced)
- **Test Preparation Course** (Completed/None)
- **Reading Score** (0-100)
- **Writing Score** (0-100)

The application features a modern, interactive web interface built with Flask and includes comprehensive data processing, model training, and prediction capabilities.

## ✨ Features

- 🤖 **Multiple ML Models**: Trains and evaluates 6 different regression algorithms
- 🎨 **Interactive UI**: Modern, responsive web interface with animations
- 📊 **Automated Pipeline**: Complete data ingestion, transformation, and training workflow
- 🔧 **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- 📝 **Comprehensive Logging**: Detailed logging system for debugging and monitoring
- 🛡️ **Error Handling**: Custom exception handling throughout the application
- 💾 **Model Persistence**: Saves trained models and preprocessors for reuse

## 📁 Project Structure

```
student-performance-predictor/
│
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation setup
│
├── src/
│   ├── __init__.py
│   ├── logger.py                   # Logging configuration
│   ├── exception.py                # Custom exception handling
│   ├── utils.py                    # Utility functions
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py  # Feature engineering & preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── predict_pipeline.py     # Prediction pipeline
│
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Interactive prediction form
│
├── Notebook/
│   └── Data/
│       └── stud.csv                # Student performance dataset
│
├── artifacts/                      # Generated artifacts
│   ├── data.csv                    # Raw data
│   ├── train.csv                   # Training data
│   ├── test.csv                    # Testing data
│   ├── model.pkl                   # Trained model
│   └── preprocessor.pkl            # Data preprocessor
│
└── logs/                           # Application logs
    └── [timestamp].log
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package

```bash
pip install -e .
```

## 💻 Usage

### Training the Model

Run the data ingestion, transformation, and model training pipeline:

```bash
python src/components/data_ingestion.py
```

This will:
1. Load the dataset from `Notebook/Data/stud.csv`
2. Split data into train/test sets (80/20)
3. Apply data transformations and preprocessing
4. Train multiple ML models with hyperparameter tuning
5. Save the best model and preprocessor to `artifacts/`

### Running the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000/`

### Making Predictions

1. Navigate to `http://localhost:5000/predictdata`
2. Fill in the student information form
3. Click "Predict Math Score"
4. View the predicted score with visual feedback

## 🔬 ML Pipeline

### 1. Data Ingestion (`data_ingestion.py`)

- Reads student performance dataset
- Splits data into training (80%) and testing (20%) sets
- Saves raw, train, and test data to artifacts folder

```python
obj = DataIngestion()
train_data, test_data = obj.initiate_data_ingestion()
```

### 2. Data Transformation (`data_transformation.py`)

**Numerical Features:**
- Reading Score
- Writing Score

**Transformation Steps:**
- Median imputation for missing values
- Standard scaling (with_mean=False)

**Categorical Features:**
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course

**Transformation Steps:**
- Most frequent imputation for missing values
- One-Hot Encoding

```python
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
```

### 3. Model Training (`model_trainer.py`)

**Models Evaluated:**
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor**
6. **AdaBoost Regressor**

**Hyperparameter Tuning:**
- GridSearchCV with 3-fold cross-validation
- Optimized parameters for each model
- Best model selection based on R² score

```python
modeltrainer = ModelTrainer()
r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
```

### 4. Prediction Pipeline (`predict_pipeline.py`)

```python
# Create custom data instance
data = CustomData(
    gender='male',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch='standard',
    test_preparation_course='none',
    reading_score=72,
    writing_score=74
)

# Get prediction
pipeline = PredictPipeline()
pred_df = data.get_data_as_data_frame()
result = pipeline.predict(pred_df)
print(f"Predicted Math Score: {result[0]}")
```

## 📊 Model Performance

The system evaluates all models and selects the best performer based on R² score on the test set. Typical performance metrics:

| Model | R² Score | Training Time |
|-------|----------|---------------|
| Gradient Boosting | ~0.88 | Medium |
| Random Forest | ~0.87 | Medium |
| XGBoost | ~0.86 | Fast |
| AdaBoost | ~0.85 | Fast |
| Decision Tree | ~0.75 | Fast |
| Linear Regression | ~0.72 | Very Fast |

*Note: Actual scores may vary based on data and hyperparameter tuning*

## 🌐 API Documentation

### Endpoints

#### GET `/`
Returns the landing page

**Response:** HTML page

#### GET `/predictdata`
Returns the prediction form

**Response:** HTML page with interactive form

#### POST `/predictdata`
Submits student data for math score prediction

**Request Body (Form Data):**
```
gender: string (male/female)
ethnicity: string (group A/B/C/D/E)
parental_level_of_education: string
lunch: string (standard/free/reduced)
test_preparation_course: string (none/completed)
reading_score: float (0-100)
writing_score: float (0-100)
```

**Response:** HTML page with predicted score

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with animations
- **JavaScript** - Interactivity

### Development Tools
- **dill** - Object serialization
- **pickle** - Model persistence
- **logging** - Application logging

## 📦 Dependencies

```txt
pandas
numpy
scikit-learn
xgboost
Flask
dill
pickle-mixin
```

## 🔧 Configuration

### Logging Configuration

Logs are stored in `logs/` directory with timestamp-based filenames:
- Format: `[timestamp] line_number module_name - level - message`
- Level: INFO
- Automatic directory creation

### Model Artifacts

All trained models and preprocessors are saved in the `artifacts/` directory:
- `model.pkl` - Best trained model
- `preprocessor.pkl` - Data preprocessing pipeline
- `train.csv`, `test.csv`, `data.csv` - Dataset splits

## 🧪 Testing

Run predictions with sample data:

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create test data
test_data = CustomData(
    gender='female',
    race_ethnicity='group C',
    parental_level_of_education='some college',
    lunch='standard',
    test_preparation_course='completed',
    reading_score=85,
    writing_score=88
)

# Get prediction
pipeline = PredictPipeline()
result = pipeline.predict(test_data.get_data_as_data_frame())
print(f"Predicted Score: {result[0]:.2f}")
```

## 🐛 Error Handling

The application includes comprehensive error handling:

- **CustomException**: Captures detailed error information including:
  - Script name
  - Line number
  - Error message
- **Logging**: All errors are logged with timestamps
- **User-Friendly Messages**: Web interface shows appropriate error messages

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Animations**: Smooth transitions and hover effects
- **Real-time Validation**: Input validation for score ranges (0-100)
- **Visual Feedback**: Progress bar showing predicted score
- **Modern Aesthetics**: Gradient backgrounds and glassmorphic design

## 🚧 Future Enhancements

- [ ] Add more features (attendance, study hours, etc.)
- [ ] Implement user authentication
- [ ] Add database for storing predictions
- [ ] Create REST API for external integrations
- [ ] Add data visualization dashboard
- [ ] Implement A/B testing for models
- [ ] Add batch prediction capability
- [ ] Deploy to cloud platform (AWS/Azure/GCP)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset source: [Student Performance Dataset]
- Inspiration from various ML education projects
- Flask documentation and community
- scikit-learn documentation

## 📞 Contact

For questions or feedback, please reach out:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ If you found this project helpful, please give it a star!

Made with ❤️ and Python
