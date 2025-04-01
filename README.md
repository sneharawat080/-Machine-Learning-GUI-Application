

# Machine Learning GUI Application

This is a GUI-based Machine Learning application built using **Streamlit** that allows users to upload datasets (CSV format) and apply various machine learning algorithms such as Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes, Support Vector Machine (SVM), Logistic Regression, and Linear Regression.

## Features

- Upload CSV dataset.
- Select target column for predictions.
- Choose from various machine learning algorithms.
- View results such as accuracy, confusion matrix, and classification report.
- Interactive visualizations like decision boundaries, scatter plots, and more.

## Requirements

1. **Python 3.x**
2. **Libraries**: 
   - Streamlit
   - Pandas
   - Numpy
   - Matplotlib
   - Seaborn
   - Scikit-learn
   - MLxtend

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ML_GUI_Project.git
   ```

2. Navigate into the project folder:
   ```bash
   cd ML_GUI_Project
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, you can manually install libraries using:
   ```bash
   pip install streamlit pandas numpy scikit-learn mlxtend matplotlib seaborn
   ```

## How to Run the Application

1. After installing all dependencies, run the app with Streamlit:
   ```bash
   streamlit run app.py
   ```

   Alternatively, you can also run it using Python directly:
   ```bash
   python -m streamlit run app.py
   ```

2. This will open a web app in your default browser at `http://localhost:8501`.

3. **Upload a dataset** in CSV format, select a target column, and choose a machine learning algorithm. The results (accuracy, confusion matrix, etc.) will be displayed.

## Dataset Example

A sample dataset is provided in the repository (`sample_dataset.csv`). You can upload your own datasets to test the application.

## Contribution

Feel free to fork this repository and submit pull requests to improve the project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
