"""Main application for ticket classification."""

import os
import streamlit as st
from classifier.model import train_model_from_csv
from preprocess.clean_text import clean_text
from utils.priority import assign_priority
from utils.confidence import format_confidence
from classifier.labels import CATEGORIES

# Page configuration
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# Title
st.title("🎫 Support Ticket Classifier")
st.markdown("Classify customer support tickets into categories with confidence scores and priority levels.")


@st.cache_resource
def load_model():
    """
    Load and cache the trained model.
    
    Returns:
        Trained TicketClassifier instance
    """
    csv_path = "data/sample_tickets.csv"
    
    if not os.path.exists(csv_path):
        st.error(f"Training data not found at {csv_path}")
        return None
    
    try:
        classifier = train_model_from_csv(csv_path)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    """Main application function."""
    # Load model
    classifier = load_model()
    
    if classifier is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("### Categories")
        for category in CATEGORIES:
            st.markdown(f"- **{category}**")
        
        st.markdown("### Priority Levels")
        st.markdown("""
        - **High**: Contains urgent keywords (refund, error, failed, etc.)
        - **Medium**: Default priority
        - **Low**: Positive sentiment detected
        """)
    
    # Main input area
    st.header("📝 Classify a Ticket")
    
    # Text input
    ticket_text = st.text_area(
        "Enter ticket text:",
        height=150,
        placeholder="Example: I was charged twice for my subscription. Can I get a refund?"
    )
    
    # Classify button
    if st.button("Classify Ticket", type="primary"):
        if not ticket_text or not ticket_text.strip():
            st.warning("Please enter ticket text before classifying.")
        else:
            try:
                # Clean text
                cleaned_text = clean_text(ticket_text)
                
                # Predict category
                category, confidence, probabilities = classifier.predict(cleaned_text)
                
                # Assign priority
                priority = assign_priority(ticket_text)
                
                # Display results
                st.success("Classification complete!")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Category", category)
                
                with col2:
                    st.metric("Confidence", format_confidence(confidence))
                
                with col3:
                    # Color code priority
                    if priority == "High":
                        st.metric("Priority", "🔴 High")
                    elif priority == "Medium":
                        st.metric("Priority", "🟡 Medium")
                    else:
                        st.metric("Priority", "🟢 Low")
                
                # Detailed breakdown
                st.subheader("📊 Detailed Results")
                
                # Category probabilities
                st.markdown("**Category Probabilities:**")
                prob_dict = {CATEGORIES[i]: float(prob) for i, prob in enumerate(probabilities)}
                
                # Sort by probability
                sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                
                for cat, prob in sorted_probs:
                    st.progress(prob, text=f"{cat}: {format_confidence(prob)}")
                
                # Priority explanation
                st.markdown("**Priority Reasoning:**")
                if priority == "High":
                    st.info("⚠️ High priority assigned due to urgent keywords detected in the ticket.")
                elif priority == "Low":
                    st.info("✅ Low priority assigned due to positive sentiment detected.")
                else:
                    st.info("ℹ️ Medium priority assigned (default).")
                
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")
    
    # Example tickets section
    with st.expander("📋 Example Tickets"):
        examples = [
            ("I was charged twice for my subscription. Can I get a refund?", "Billing"),
            ("The login page is not working. I keep getting an error message.", "Technical"),
            ("I want to update my email address on my account.", "Account"),
            ("Thank you so much for the excellent service!", "Feedback"),
        ]
        
        for example_text, expected_category in examples:
            if st.button(f"Try: {example_text[:50]}...", key=f"example_{example_text[:10]}"):
                st.session_state.example_text = example_text
        
        if 'example_text' in st.session_state:
            ticket_text = st.text_area(
                "Enter ticket text:",
                value=st.session_state.example_text,
                height=150,
                key="example_input"
            )


if __name__ == "__main__":
    main()

