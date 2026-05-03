import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ML Experiment Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DATA LOADING AND CACHING ---


@st.cache_data
def load_data(file_path="runs_summary.json"):
    """
    Load the experiment summary JSON file into a Pandas DataFrame.
    The @st.cache_data decorator ensures the data is loaded only once.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at '{
                 file_path}'. Please run `parse_runs.py` first.")
        return None
    try:
        df = pd.read_json(file_path)
        # Add a 'num_features' column for easier filtering and analysis
        df['num_features'] = df['features_used'].apply(len)
        return df
    except Exception as e:
        st.error(f"Failed to load or parse data file: {e}")
        return None

# --- HELPER FUNCTIONS ---


def display_confusion_matrix(cm_data):
    """
    Displays the confusion matrix in a formatted table.
    Input: cm_data is a 2x2 list like [[TN, FP], [FN, TP]]
    """
    tn, fp, fn, tp = cm_data[0][0], cm_data[0][1], cm_data[1][0], cm_data[1][1]

    cm_df = pd.DataFrame(
        data=[[tn, fp], [fn, tp]],
        columns=["Predicted Healthy", "Predicted Faulted"],
        index=["True Healthy", "True Faulted"]
    )
    st.table(cm_df)


def display_classification_report(report_data):
    """
    Displays the structured classification report.
    """
    # Convert dict to a DataFrame for prettier display
    df = pd.DataFrame(report_data).transpose()
    # Format float columns for better readability
    float_cols = ['precision', 'recall', 'f1-score']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
    st.table(df)

# --- MAIN APP LOGIC ---


def main():
    st.title("🧪 ML Experiment Dashboard")
    st.markdown(
        "A dashboard to track and compare fault detection model experiments.")

    df = load_data()

    if df is None:
        return  # Stop execution if data loading failed

    # --- SIDEBAR FOR FILTERS ---
    st.sidebar.header("Filter Experiments")

    # Filter by Test Accuracy
    min_test_acc = st.sidebar.slider(
        "Minimum Test Accuracy",
        min_value=0.0,
        max_value=1.0,
        value=0.70,  # Default value
        step=0.01,
    )

    # Filter by Features Used
    # Get a unique, sorted list of all features used across all runs
    all_features = sorted(
        list(set(f for features_list in df['features_used'] for f in features_list)))
    selected_features = st.sidebar.multiselect(
        "Filter by Features Contained",
        options=all_features,
        help="Show runs that include ALL of the selected features."
    )

    # Apply filters to the DataFrame
    filtered_df = df[df['test_results'].apply(
        lambda x: x.get('accuracy', 0.0) >= min_test_acc)]

    if selected_features:
        # This logic ensures that a run must contain ALL selected features to be shown
        filtered_df = filtered_df[
            filtered_df['features_used'].apply(lambda features: all(
                f in features for f in selected_features))
        ]

    # --- MAIN VIEW ---
    st.header("Master Runs Table")

    if filtered_df.empty:
        st.warning("No runs match the current filter settings.")
    else:
        # Define columns to display in the main table for brevity
        summary_cols = {
            'run_codename': "Run Codename",
            'test_accuracy': "Test Acc",
            'last_val_acc': "Val Acc",
            'num_features': "# Features",
            'last_train_loss': "Train Loss"
        }
        # Extract test accuracy for display and sorting
        display_df = filtered_df.copy()
        display_df['test_accuracy'] = display_df['test_results'].apply(
            lambda x: x.get('accuracy', 0.0))

        st.dataframe(
            display_df[list(summary_cols.keys())],
            column_config={
                "test_accuracy": st.column_config.NumberColumn(format="%.4f"),
                "last_val_acc": st.column_config.NumberColumn(format="%.4f"),
                "last_train_loss": st.column_config.NumberColumn(format="%.4f"),
            },
            use_container_width=True
        )

    st.markdown("---")
    st.header("Detailed Run View")

    # Dropdown to select a specific run from the (filtered) list
    # The list is sorted so the newest runs appear first
    run_options = filtered_df.sort_values(
        by='run_codename', ascending=False)['run_codename']
    selected_run_codename = st.selectbox(
        "Select a run to inspect:",
        options=run_options
    )

    if not selected_run_codename:
        st.info("Select a run from the dropdown to see its details.")
        return

    # Get the full data for the selected run
    selected_run_data = filtered_df[filtered_df['run_codename']
                                    == selected_run_codename].iloc[0]

    # --- DISPLAY DETAILED INFORMATION ---

    # 1. Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Test Accuracy",
            f"{selected_run_data['test_results'].get('accuracy', 0.0):.4f}"
        )
    with col2:
        st.metric(
            "Validation Accuracy",
            f"{selected_run_data['last_val_acc']:.4f}"
        )
    with col3:
        st.metric(
            "Validation Loss",
            f"{selected_run_data['last_val_loss']:.4f}"
        )

    # 2. Configuration
    st.subheader("Configuration")
    st.write("**Features Used:**")
    st.info(f"`{selected_run_data['features_used']}`")

    # 3. Visuals (Plots and Confusion Matrix)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Curves")
        if os.path.exists(selected_run_data['training_curves_path']):
            st.image(selected_run_data['training_curves_path'])
        else:
            st.warning("Training curves image not found.")

    with col2:
        st.subheader("Confusion Matrix")
        if selected_run_data['confusion_matrix']:
            display_confusion_matrix(selected_run_data['confusion_matrix'])
        else:
            st.warning("No confusion matrix data found.")

    # 4. Classification Report
    st.subheader("Classification Report")
    if selected_run_data['test_results']:
        display_classification_report(selected_run_data['test_results'])
    else:
        st.warning("No classification report data found.")


if __name__ == "__main__":
    main()
