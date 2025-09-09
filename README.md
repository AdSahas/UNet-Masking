# Machine Learning Portfolio Projects

This repository showcases multiple machine learning projects demonstrating various techniques and applications. Each project includes comprehensive analysis, code implementation, and detailed results.

## 🚀 Live Portfolio

Visit the portfolio website: [Open index.html in your browser](./index.html)

## 📊 Featured Projects

### 1. Classification Project - Customer Churn Prediction
- **Objective**: Predict customer churn for a telecommunications company
- **Techniques**: Logistic Regression, Random Forest, SVM
- **Key Results**: 85.2% accuracy with Random Forest
- **Location**: `projects/classification/`
- **Files**: `classification_analysis.py`, `index.html`

### 2. Regression Project - House Price Prediction  
- **Objective**: Predict real estate prices using property features
- **Techniques**: Linear Regression, Random Forest, Gradient Boosting
- **Key Results**: RMSE of $23,450 with Gradient Boosting
- **Location**: `projects/regression/`
- **Files**: `regression_analysis.py`, `index.html`

### 3. Clustering Project - Customer Segmentation
- **Objective**: Identify distinct customer groups for targeted marketing
- **Techniques**: K-Means, Hierarchical Clustering, DBSCAN
- **Key Results**: 5 distinct customer segments identified
- **Location**: `projects/clustering/`
- **Files**: `clustering_analysis.py`, `index.html`

## 🛠 Technologies Used

- **Python**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **HTML/CSS/JavaScript**: Portfolio website

## 📁 Repository Structure

```
PortfolioProjects/
├── index.html                 # Main portfolio page
├── styles.css                 # Main stylesheet
├── script.js                  # Portfolio interactions
├── projects/
│   ├── classification/
│   │   ├── index.html         # Classification project page
│   │   ├── classification_analysis.py
│   │   └── project-styles.css
│   ├── regression/
│   │   ├── index.html         # Regression project page
│   │   └── regression_analysis.py
│   └── clustering/
│       ├── index.html         # Clustering project page
│       └── clustering_analysis.py
└── README.md
```

## 🚀 Getting Started

### View the Portfolio
1. Clone this repository
2. Open `index.html` in your web browser
3. Navigate through the different ML projects

### Run the Python Projects
1. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Run any project:
   ```bash
   # Classification project
   cd projects/classification/
   python classification_analysis.py
   
   # Regression project  
   cd projects/regression/
   python regression_analysis.py
   
   # Clustering project
   cd projects/clustering/
   python clustering_analysis.py
   ```

## 📈 Project Highlights

### Classification (Customer Churn)
- **Dataset**: 7,043 customers with 21 features
- **Best Model**: Random Forest (85.2% accuracy, F1: 0.78, AUC: 0.89)
- **Key Insights**: Monthly charges and contract type are top churn predictors

### Regression (House Prices)
- **Dataset**: 1,460 houses with 79 features
- **Best Model**: Gradient Boosting (RMSE: $23,450, R²: 0.912)
- **Key Insights**: Overall quality and total square footage drive price

### Clustering (Customer Segmentation)
- **Dataset**: 2,000 customers with RFM analysis
- **Best Model**: K-Means with 5 clusters (Silhouette: 0.67)
- **Key Insights**: 5 distinct segments: Champions, Loyal, Potential, At Risk, Price Sensitive

## 🎯 Key Features

- **Interactive Portfolio**: Easy navigation between projects
- **Comprehensive Analysis**: Each project includes EDA, modeling, and evaluation
- **Visual Results**: Charts and plots for better understanding
- **Practical Applications**: Real-world business use cases
- **Code Quality**: Well-documented, modular Python code
- **Responsive Design**: Works on desktop and mobile devices

## 🔧 Adding New Projects

To add a new ML project:

1. Create a new directory under `projects/`
2. Create an `index.html` page following the existing template
3. Add your Python analysis script
4. Update the main `index.html` to include your project card
5. Test the navigation and links

## 📧 Contact

This portfolio demonstrates practical machine learning skills across classification, regression, and clustering problems. Each project showcases the complete ML pipeline from data exploration to model deployment.

---
*Built with ❤️ to showcase machine learning expertise*
