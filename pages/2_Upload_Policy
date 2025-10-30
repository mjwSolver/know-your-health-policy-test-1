import streamlit as st
from snowflake.snowpark.context import get_active_session
import io
import re

# --- Utility Function for Security ---
def sanitize_path_segment(input_str):
    """
    Sanitizes user input to be safe for use as a path segment in a stage.
    Removes leading/trailing whitespace and potentially dangerous characters 
    like path separators (/, \, ..).
    """
    if not input_str:
        return ""
    
    # 1. Remove leading/trailing whitespace
    cleaned_str = input_str.strip()
    
    # 2. Replace known path-breaking characters with an underscore
    # This prevents directory traversal (e.g., '..') and other path issues
    cleaned_str = re.sub(r'[\\/:*?"<>|.]', '_', cleaned_str)
    
    # 3. Ensure the string is not empty after cleaning
    return cleaned_str if cleaned_str else "default"

# --- Page Configuration ---
st.set_page_config(
    page_title="Policy Ingestion",
    page_icon="📄"
)

st.title("📄 Policy Ingestion")
st.write("Upload a new policy PDF to the `@DOCUMENT_STAGE_2`. This will trigger the automated ingestion pipeline.")

try:
    session = get_active_session()
except Exception as e:
    st.error("Could not get active Snowpark session. This app must be run in Snowflake.")
    st.stop()

# --- UI for Metadata ---
st.header("1. Policy Metadata")
st.info("This metadata will be used to create the folder structure in the stage.", icon="ℹ️")

insurer = st.text_input("Insurer Name", placeholder="e.g., Flakeseed")
policy_name = st.text_input("Policy File Name (must end in .pdf)", placeholder="e.g., Health_Gold_2025.pdf")

# --- UI for File Upload ---
st.header("2. Upload File")
uploaded_file = st.file_uploader(
    "Upload Policy PDF", 
    type="pdf", 
    key="policy_uploader"
)

# --- Upload Logic ---
st.header("3. Execute Upload")
if st.button("Upload to Stage", type="primary"):
    
    if not insurer:
        st.warning("Please enter an Insurer Name.")
    elif not policy_name or not policy_name.lower().endswith('.pdf'):
        st.warning("Please enter a valid Policy File Name ending in .pdf.")
    elif uploaded_file is None:
        st.warning("Please upload a PDF file.")
    else:

        safe_insurer = sanitize_path_segment(insurer)
        
        try:
            base_name, extension = policy_name.rsplit('.', 1)
            safe_base_name = sanitize_path_segment(base_name)
            safe_policy_name = f"{safe_base_name}.{extension}"
            
        except ValueError:
            safe_policy_name = sanitize_path_segment(policy_name)
        
        target_stage_path = f"@DOCUMENT_STAGE_2/{safe_insurer}/{safe_policy_name}"
        
        with st.spinner(f"Uploading file to {target_stage_path}..."):
            try:
                file_bytes = uploaded_file.getvalue()
                
                file_stream = io.BytesIO(file_bytes)
                
                session.file.put_stream(
                    input_stream=file_stream, 
                    stage_location=target_stage_path, 
                    auto_compress=False, 
                    overwrite=True
                )
                
                st.success(f"✅ Successfully uploaded to: {target_stage_path}")
                st.info("The automated ingestion task will process this file shortly.")

                # st.session_state['policy_uploader'] = None
                # st.rerun()
                
            except Exception as e:
                st.error(f"Error uploading file: {e}")