import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
import plotly.express as px

# --- Load Data ---
# Define the URL for the dataset
DATA_URL = "https://raw.githubusercontent.com/adinplb/LR-SVM-NN_Breast-Tumor_Prediction/refs/heads/main/dataset/Breast%20Cancer%20Wisconsin.csv"
# Define the column names for the dataset
FEATURE_NAMES = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                 "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
                 "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
                 "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
                 "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                 "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                 "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]


@st.cache_data
def load_data():
    """
    Loads the breast cancer dataset from the specified URL,
    drops unnecessary columns, and maps diagnosis to numerical values.
    Uses Streamlit's caching to avoid reloading data on every rerun.
    """
    df = pd.read_csv(DATA_URL, names=FEATURE_NAMES, header=0)
    # Drop 'id' and 'Unnamed: 32' columns. 'errors='ignore'' prevents error if 'Unnamed: 32' is not found.
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    # Map 'M' (Malignant) to 1 and 'B' (Benign) to 0 for the 'diagnosis' column.
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# Load the dataset when the script runs
df = load_data()


# --- Preprocessing Functions ---
def preprocess_data(df):
    """
    Performs data preprocessing steps:
    1. Outlier detection and removal using the IQR method.
    2. Feature engineering using PCA for highly correlated features.
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    st.subheader("Outlier Handling")
    st.write("Removing outliers using the Interquartile Range (IQR) method.")
    initial_shape = df.shape[0]
    # Iterate through numerical columns to detect and remove outliers
    for col in df.select_dtypes(include=np.number).columns:
        if col != 'diagnosis':  # Exclude the target variable from outlier removal
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Filter out rows where values are outside the IQR range
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    st.write(f"Initial number of samples: {initial_shape}")
    st.write(f"Number of samples after outlier removal: {df.shape[0]}")

    st.subheader("Feature Engineering with PCA")
    st.write("Applying Principal Component Analysis (PCA) to reduce dimensionality of highly correlated features.")
    # Define highly correlated features for PCA
    correlated_features = ['radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst',
                           'perimeter_worst', 'area_worst']
    # Check if all correlated features exist in the DataFrame
    if all(col in df.columns for col in correlated_features):
        pca = PCA(n_components=1, random_state=123) # Initialize PCA with 1 component
        # Fit PCA and transform the correlated features into a single 'dimension'
        df['dimension'] = pca.fit_transform(df[correlated_features]).flatten()
        # Drop the original correlated features
        df = df.drop(columns=correlated_features)
        st.write(f"Created 'dimension' feature from: {', '.join(correlated_features)}")
    else:
        st.write("Skipping PCA: Not all correlated features found in the dataset.")

    return df


def split_and_scale(df):
    """
    Splits the data into training and testing sets and scales numerical features.
    """
    st.subheader("Data Splitting and Scaling")
    # Separate features (X) and target (y)
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numerical features for scaling
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler() # Initialize StandardScaler
    # Fit scaler on training data and transform both training and test data
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    st.write("Data split into training and testing sets (80/20).")
    st.write("Numerical features scaled using StandardScaler.")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """
    Applies SMOTE to balance the training dataset.
    """
    st.subheader("SMOTE for Class Balancing")
    st.write("Applying Synthetic Minority Over-sampling Technique (SMOTE) to balance the training data.")
    # Count class distribution before SMOTE
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    st.write("Class distribution before SMOTE (Training Set):")
    st.write(pd.DataFrame({'Class': unique_train, 'Count': counts_train}))

    smote = SMOTE(random_state=42) # Initialize SMOTE
    # Resample the training data
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Count class distribution after SMOTE
    unique_train_resampled, counts_train_resampled = np.unique(y_train_resampled, return_counts=True)
    st.write("Class distribution after SMOTE (Training Set):")
    st.write(pd.DataFrame({'Class': unique_train_resampled, 'Count': counts_train_resampled}))

    return X_train_resampled, y_train_resampled


# --- Model Training and Evaluation Functions ---
def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, use_smote=False):
    """
    Trains and evaluates a Logistic Regression model.
    Optionally applies SMOTE to the training data.
    """
    st.write("Training Logistic Regression Model...")
    X_train_processed, y_train_processed = X_train.copy(), y_train.copy()
    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train, y_train)

    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Use 'liblinear' solver and balanced class weight
    model.fit(X_train_processed, y_train_processed)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, cm, report


def train_and_evaluate_nn(X_train, X_test, y_train, y_test, use_smote=False):
    """
    Trains and evaluates a Neural Network model.
    Optionally applies SMOTE to the training data.
    """
    st.write("Training Neural Network Model...")
    X_train_processed, y_train_processed = X_train.copy(), y_train.copy()
    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train, y_train)

    # Define the neural network architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)), # Input layer with 128 neurons
        Dense(64, activation='relu'), # Hidden layer with 64 neurons
        Dense(1, activation='sigmoid') # Output layer with 1 neuron for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model

    # Train the model (verbose=0 to suppress extensive output during training)
    model.fit(X_train_processed, y_train_processed, epochs=10, batch_size=32, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0) # Predict probabilities
    y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, cm, report


def train_and_evaluate_svm(X_train, X_test, y_train, y_test, use_smote=False):
    """
    Trains and evaluates an SVM model.
    Optionally applies SMOTE to the training data.
    """
    st.write("Training SVM Model...")
    X_train_processed, y_train_processed = X_train.copy(), y_train.copy()
    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train, y_train)

    model = SVC(kernel='rbf', random_state=42, class_weight='balanced') # Use RBF kernel and balanced class weight
    model.fit(X_train_processed, y_train_processed)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, cm, report


# --- Visualization Functions ---
def plot_confusion_matrix(cm, class_names, title):
    """
    Generates and displays a Plotly heatmap for a confusion matrix.
    """
    fig = px.imshow(cm,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=class_names,
                    y=class_names,
                    color_continuous_scale=px.colors.sequential.Blues,
                    text_auto=True, # Automatically display values in cells
                    title=title)
    fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
    st.plotly_chart(fig, use_container_width=True) # Use container width for better responsiveness


def plot_classification_report(report, title):
    """
    Generates and displays a Plotly bar chart for a classification report.
    """
    # Convert classification report dictionary to a DataFrame for plotting
    report_df = pd.DataFrame(report).transpose()
    # Remove 'accuracy' row before melting for better visualization of per-class metrics
    if 'accuracy' in report_df.index:
        accuracy_val = report_df.loc['accuracy', 'f1-score'] # Accuracy is typically in f1-score column for overall
        report_df = report_df.drop('accuracy')
    # Melt the DataFrame for Plotly's bar chart
    report_df_melted = report_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    # Filter for precision, recall, f1-score, and support
    report_df_melted = report_df_melted[report_df_melted['Metric'].isin(['precision', 'recall', 'f1-score', 'support'])]

    fig = px.bar(report_df_melted,
                 x='Metric',
                 y='Score',
                 color='index', # Color by class (0, 1, or macro avg, weighted avg)
                 barmode='group',
                 title=title,
                 labels={'Score': 'Score', 'index': 'Class/Average', 'Metric': 'Metric'},
                 color_discrete_sequence=px.colors.qualitative.Dark24,
                 height=400)
    # Add accuracy as a separate text annotation if available
    if 'accuracy_val' in locals():
        fig.add_annotation(
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            text=f"Overall Accuracy: {accuracy_val:.2f}",
            showarrow=False,
            font=dict(size=14, color="black")
        )
    fig.update_yaxes(range=[0, 1.1]) # Ensure scores are within 0-1 range
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_distribution(df, feature, title):
    """
    Generates and displays a Plotly histogram and box plot for feature distribution,
    colored by diagnosis.
    """
    fig = px.histogram(df, x=feature, color='diagnosis', marginal='box',
                       title=title,
                       color_discrete_sequence=['#4CAF50', '#FF6347'], # Green for Benign (0), Red for Malignant (1)
                       hover_data=df.columns)
    fig.update_layout(bargap=0.1) # Add gap between bars for better readability
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(df, title):
    """
    Generates and displays a Plotly heatmap for the correlation matrix.
    """
    corr_matrix = df.corr().round(2) # Calculate correlation matrix and round to 2 decimal places
    fig = px.imshow(corr_matrix,
                    color_continuous_scale=px.colors.diverging.RdBu, # Diverging color scale
                    text_auto=True, # Display correlation values
                    aspect="auto", # Adjust aspect ratio
                    title=title)
    fig.update_layout(height=800, width=800) # Set fixed size for better readability
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit App ---
def main():
    """
    Main function to run the Streamlit Breast Cancer Prediction App.
    Sets up the UI, handles data flow, and displays results.
    """
    st.set_page_config(layout="wide", page_title="Breast Tumor Prediction", page_icon="🎗️")

    st.title("🎗️ Breast Tumor Prediction and Diagnosis")
    st.markdown("""
        This interactive web application helps predict breast tumor diagnosis (Benign or Malignant)
        using quantitative cell nuclear phenotype features and various supervised machine learning algorithms.
        Explore the data, understand the preprocessing steps, and compare model performances.
    """)

    st.sidebar.header("Navigation & Settings")
    st.sidebar.markdown("Use the sections below to navigate and configure the analysis.")

    # --- Data Loading and Preprocessing ---
    with st.sidebar.expander("Data Loading"):
        st.write("Dataset loaded from GitHub.")
        st.write("Original dataset includes 32 features and `id` column.")
        st.write("`id` and `Unnamed: 32` columns are dropped.")
        st.write("`diagnosis` column mapped: 'M' (Malignant) -> 1, 'B' (Benign) -> 0.")

    # Display raw data info
    with st.expander("Raw Data Overview & Information"):
        st.header("Raw Data Overview")
        st.markdown("### First 5 Rows of Raw Data")
        st.dataframe(df.head())
        st.markdown("### Data Information")
        st.write("This provides a summary of the DataFrame including column data types, non-null values, and memory usage.")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s) # Display df.info() output
        st.markdown("### Descriptive Statistics")
        st.write("Summary statistics for numerical columns, including count, mean, std, min, max, and quartiles.")
        st.dataframe(df.describe().round(2))
        st.markdown("### Missing Values and Duplicates")
        st.write(f"Number of duplicated rows: **{df.duplicated().sum()}**")
        st.write("Missing values per column:")
        st.dataframe(df.isna().sum().to_frame(name='Missing Count'))


    st.header("Data Preprocessing Steps")
    st.markdown("This section details the steps taken to clean and prepare the data for model training.")
    preprocessed_df = preprocess_data(df.copy()) # Pass a copy to preprocessing to avoid modifying the original 'df'

    # Display preprocessed data info
    with st.expander("Preprocessed Data Overview"):
        st.markdown("### First 5 Rows of Preprocessed Data")
        st.dataframe(preprocessed_df.head())
        st.markdown("### Preprocessed Data Information")
        buffer = StringIO()
        preprocessed_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.markdown("### Preprocessed Data Descriptive Statistics")
        st.dataframe(preprocessed_df.describe().round(2))

    X_train, X_test, y_train, y_test = split_and_scale(preprocessed_df)


    # --- Exploratory Data Analysis ---
    with st.expander("Exploratory Data Analysis (EDA)"):
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizations to understand the distribution and relationships within the preprocessed data.")

        st.subheader("Feature Distributions by Diagnosis")
        st.write("Observe how each feature's distribution varies between benign (0) and malignant (1) diagnoses.")
        selected_eda_feature = st.selectbox(
            "Select a feature to view its distribution:",
            [col for col in preprocessed_df.columns if col not in ['diagnosis']]
        )
        if selected_eda_feature:
            plot_feature_distribution(preprocessed_df, selected_eda_feature, f"Distribution of {selected_eda_feature} by Diagnosis")

        st.subheader("Correlation Matrix")
        st.write("A heatmap showing the correlation coefficients between all numerical features. Values closer to 1 or -1 indicate stronger correlation.")
        plot_correlation_matrix(preprocessed_df.drop(columns=['diagnosis'], errors='ignore'), "Correlation Matrix of Preprocessed Features")


    # --- Model Training and Evaluation ---
    st.header("Machine Learning Model Training & Evaluation")
    st.markdown("Select a machine learning model and whether to apply SMOTE for class balancing, then view its performance metrics.")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Neural Network", "SVM"])
    with col2:
        use_smote = st.checkbox("Apply SMOTE for class balancing", value=True, help="SMOTE (Synthetic Minority Over-sampling Technique) helps address imbalanced datasets by creating synthetic samples of the minority class.")

    st.markdown("---") # Separator

    model = None
    cm = None
    report = None

    # Dynamically call the training function based on model choice
    if model_choice == "Logistic Regression":
        st.subheader("Logistic Regression Model Performance")
        model, cm, report = train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, use_smote)
    elif model_choice == "Neural Network":
        st.subheader("Neural Network Model Performance")
        model, cm, report = train_and_evaluate_nn(X_train, X_test, y_train, y_test, use_smote)
    elif model_choice == "SVM":
        st.subheader("Support Vector Machine (SVM) Model Performance")
        model, cm, report = train_and_evaluate_svm(X_train, X_test, y_train, y_test, use_smote)

    if cm is not None and report is not None:
        st.markdown("### Confusion Matrix")
        st.write("A confusion matrix shows the number of correct and incorrect predictions made by the classification model, compared to the actual outcomes (true labels).")
        plot_confusion_matrix(cm, ['Benign (0)', 'Malignant (1)'], f"{model_choice} Confusion Matrix")

        st.markdown("### Classification Report")
        st.write("The classification report shows the precision, recall, F1-score, and support for each class, as well as overall accuracy.")
        plot_classification_report(report, f"{model_choice} Classification Report")
    else:
        st.info("Select a model and click 'Run Analysis' (if any) to see results.")

    st.markdown("---")
    st.markdown("Created by **Muhammad Adin Palimbani** | [GitHub Project](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/blob/main/Proyek%20Pertama/README.md)")


# Required for df.info() to capture output
from io import StringIO

if __name__ == "__main__":
    main()
