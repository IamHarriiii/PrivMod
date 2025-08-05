# visualization/dashboard.py (COMPLETELY FIXED)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set page config
st.set_page_config(page_title="FL Moderation Dashboard", layout="wide")

st.title("📊 Federated Learning Moderation Dashboard")
st.markdown("Visualizing performance of decentralized content moderation system.")

# Sidebar for navigation
st.sidebar.header("Select Visualization")
option = st.sidebar.selectbox(
    "Choose Metric",
    (
        "Accuracy Over Rounds",
        "Communication Overhead", 
        "Privacy Leakage",
        "Model Evaluation Metrics",
        "Run Inference"
    )
)

# Sample data loading functions with error handling
def load_accuracy_data():
    log_path = os.path.join(project_root, "logs", "accuracy_log.csv")
    try:
        if os.path.exists(log_path):
            return pd.read_csv(log_path)
        else:
            st.info("Using sample accuracy data (actual logs not found)")
            return pd.DataFrame({
                'round': range(1, 6),
                'accuracy': [0.75, 0.80, 0.83, 0.85, 0.87]
            })
    except Exception as e:
        st.warning(f"Error loading accuracy data: {e}")
        return pd.DataFrame({'round': [1], 'accuracy': [0.8]})

def load_communication_data():
    log_path = os.path.join(project_root, "logs", "comm_overhead.csv")
    try:
        if os.path.exists(log_path):
            return pd.read_csv(log_path)
        else:
            st.info("Using sample communication data (actual logs not found)")
            return pd.DataFrame({
                'round': range(1, 6),
                'size_mb': [4.2, 4.1, 4.3, 4.0, 4.2]
            })
    except Exception as e:
        st.warning(f"Error loading communication data: {e}")
        return pd.DataFrame({'round': [1], 'size_mb': [4.0]})

def load_dp_data():
    log_path = os.path.join(project_root, "logs", "dp_budget.json")
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                return json.load(f)
        else:
            st.info("Using sample privacy data (actual logs not found)")
            return {"epsilon": [1.2, 1.5, 1.8, 2.1, 2.3]}
    except Exception as e:
        st.warning(f"Error loading privacy data: {e}")
        return {"epsilon": [1.5, 1.8, 2.0]}

def load_evaluation_data():
    log_path = os.path.join(project_root, "logs", "evaluation_report.json")
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                return json.load(f)
        else:
            st.info("Using sample evaluation data (actual logs not found)")
            return {
                "precision": {"0": 0.9, "1": 0.85},
                "recall": {"0": 0.88, "1": 0.87},
                "f1-score": {"0": 0.89, "1": 0.86}
            }
    except Exception as e:
        st.warning(f"Error loading evaluation data: {e}")
        return {"precision": {"0": 0.85, "1": 0.82}}

# Load data with error handling
try:
    acc_df = load_accuracy_data()
    comm_df = load_communication_data()
    dp_dict = load_dp_data()
    eval_dict = load_evaluation_data()
except Exception as e:
    st.error(f"Failed to load dashboard data: {e}")
    st.stop()

# Visualization sections
if option == "Accuracy Over Rounds":
    st.subheader("📈 Model Accuracy Across FL Rounds")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=acc_df, x="round", y="accuracy", marker="o", ax=ax, linewidth=2, markersize=8)
        ax.set_xlabel("Federated Learning Round", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Model Accuracy Over FL Rounds", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.metric("Current Accuracy", f"{acc_df['accuracy'].iloc[-1]:.2%}")
        st.metric("Improvement", f"+{(acc_df['accuracy'].iloc[-1] - acc_df['accuracy'].iloc[0]):.2%}")
        st.metric("Best Round", f"Round {acc_df.loc[acc_df['accuracy'].idxmax(), 'round']}")

elif option == "Communication Overhead":
    st.subheader("📦 Communication Overhead Per Round")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(data=comm_df, x="round", y="size_mb", palette="viridis", ax=ax)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Model Update Size (MB)", fontsize=12)
        ax.set_title("Communication Overhead", fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        avg_size = comm_df['size_mb'].mean()
        total_size = comm_df['size_mb'].sum()
        st.metric("Avg Size", f"{avg_size:.1f} MB")
        st.metric("Total Transfer", f"{total_size:.1f} MB")
        st.metric("Efficiency", "High" if avg_size < 5 else "Medium")

elif option == "Privacy Leakage":
    st.subheader("🛡️ Privacy Budget (ε) Over Rounds")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        epsilons = dp_dict["epsilon"]
        rounds = list(range(1, len(epsilons)+1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=rounds, y=epsilons, marker="o", ax=ax, linewidth=2, markersize=8, color='red')
        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("Epsilon (ε)", fontsize=12)
        ax.set_title("Privacy Budget Consumption", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add privacy level annotations
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Strong Privacy')
        ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Moderate Privacy')
        ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Weak Privacy')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        current_epsilon = epsilons[-1]
        privacy_level = "Strong" if current_epsilon < 1.0 else "Moderate" if current_epsilon < 2.0 else "Weak"
        st.metric("Current ε", f"{current_epsilon:.2f}")
        st.metric("Privacy Level", privacy_level)
        remaining_budget = max(0, 3.0 - current_epsilon)
        st.metric("Remaining Budget", f"{remaining_budget:.2f}")

elif option == "Model Evaluation Metrics":
    st.subheader("🔍 Classification Report")
    
    # Create evaluation dataframe
    eval_df = pd.DataFrame(eval_dict).T
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            eval_df.style.background_gradient(cmap="Blues").format("{:.3f}"),
            use_container_width=True
        )
    
    with col2:
        # Calculate overall metrics
        if "precision" in eval_dict:
            avg_precision = np.mean(list(eval_dict["precision"].values()))
            st.metric("Avg Precision", f"{avg_precision:.3f}")
        
        if "recall" in eval_dict:
            avg_recall = np.mean(list(eval_dict["recall"].values()))
            st.metric("Avg Recall", f"{avg_recall:.3f}")
        
        if "f1-score" in eval_dict:
            avg_f1 = np.mean(list(eval_dict["f1-score"].values()))
            st.metric("Avg F1-Score", f"{avg_f1:.3f}")
    
    # Confusion matrix visualization
    st.subheader("Confusion Matrix")
    cm_path = os.path.join(project_root, "logs", "confusion_matrix.csv")
    
    try:
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path, header=None).values
        else:
            # Create sample confusion matrix
            cm = np.array([[85, 15], [10, 90]])
            st.info("Using sample confusion matrix (actual data not found)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                   xticklabels=['Not Toxic', 'Toxic'], 
                   yticklabels=['Not Toxic', 'Toxic'])
        ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not load confusion matrix: {e}")

elif option == "Run Inference":
    st.subheader("🧪 Try Your Own Input")
    st.markdown("Test the content moderation model with your own text input.")
    
    # Text input
    user_input = st.text_area(
        "Enter text to check for moderation flag:", 
        placeholder="Type your message here...",
        height=100
    )
    
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing content..."):
                try:
                    # Simple mock inference (replace with actual model)
                    # For now, we'll simulate based on keyword detection
                    toxic_keywords = ['hate', 'stupid', 'idiot', 'kill', 'die', 'horrible']
                    
                    # Calculate mock toxicity score
                    text_lower = user_input.lower()
                    toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_lower)
                    base_score = min(toxic_count * 0.3, 0.9)
                    
                    # Add some randomness for demo
                    import random
                    prob = max(0.1, min(0.9, base_score + random.uniform(-0.2, 0.2)))
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Toxicity Score", f"{prob:.3f}")
                    
                    with col2:
                        confidence = abs(prob - 0.5) * 2
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    with col3:
                        verdict = "🔴 Flagged" if prob > 0.5 else "🟢 Safe"
                        st.metric("Status", verdict)
                    
                    # Progress bar for toxicity score
                    st.subheader("Toxicity Analysis")
                    st.progress(prob)
                    
                    if prob > 0.7:
                        st.error("⚠️ High toxicity detected - Content likely to be flagged")
                    elif prob > 0.5:
                        st.warning("⚠️ Moderate toxicity detected - Content may be flagged")
                    else:
                        st.success("✅ Content appears safe")
                    
                    # Show explanation
                    st.subheader("Analysis Details")
                    if toxic_count > 0:
                        st.write(f"⚠️ Detected {toxic_count} potentially problematic pattern(s)")
                    else:
                        st.write("✅ No obvious toxic patterns detected")
                    
                    st.info("**Note**: This is a demo using keyword-based detection. The actual federated model would provide more sophisticated analysis.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.info("Make sure your federated learning system is running and models are trained.")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Sample texts for testing
    st.subheader("📝 Try These Sample Texts")
    samples = [
        "I love spending time with my family on weekends.",
        "This movie is absolutely terrible and stupid.",
        "Thank you for your help, I really appreciate it!",
        "I hate this product, it's completely useless."
    ]
    
    for i, sample in enumerate(samples):
        if st.button(f"📋 Use Sample {i+1}", key=f"sample_{i}"):
            st.session_state.sample_text = sample
            st.experimental_rerun()

# Sidebar export functionality
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Export")

if st.sidebar.button("📥 Export All Data"):
    try:
        # Create exports directory
        export_dir = os.path.join(project_root, "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Export data
        acc_df.to_csv(os.path.join(export_dir, "accuracy_data.csv"), index=False)
        comm_df.to_csv(os.path.join(export_dir, "communication_data.csv"), index=False)
        
        with open(os.path.join(export_dir, "privacy_data.json"), 'w') as f:
            json.dump(dp_dict, f, indent=2)
        
        with open(os.path.join(export_dir, "evaluation_data.json"), 'w') as f:
            json.dump(eval_dict, f, indent=2)
        
        st.sidebar.success("✅ Data exported to exports/ folder!")
        
    except Exception as e:
        st.sidebar.error(f"❌ Export failed: {e}")

# System status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Status")

# Check if FL system is running (mock check)
import requests
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ Backend API: Online")
    else:
        st.sidebar.warning("⚠️ Backend API: Issues")
except:
    st.sidebar.error("❌ Backend API: Offline")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🚀 Built with Streamlit\n💡 FL Moderation System v1.0")

# Add some custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.stProgress .st-bo {
    background-color: #ff4b4b;
}

.success-text {
    color: #00c851;
    font-weight: bold;
}

.warning-text {
    color: #ffbb33;
    font-weight: bold;
}

.error-text {
    color: #ff4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)