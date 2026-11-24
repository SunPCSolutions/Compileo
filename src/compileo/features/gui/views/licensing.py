import streamlit as st
from ..services.api_client import api_client

def render_licensing_page():
    """Renders the license activation page."""
    st.title("🔑 License Activation")
    st.markdown("""
    Welcome to Compileo! To use the software, please activate your installation with a valid license key.
    You can obtain a license key at [notin.ai](https://www.notin.ai).
    """)
    
    with st.form("activation_form"):
        license_key = st.text_input("License Key", placeholder="XXXX-XXXX-XXXX-XXXX")
        submit = st.form_submit_button("Activate Compileo", type="primary")
        
        if submit:
            if not license_key:
                st.error("Please enter a license key.")
            else:
                try:
                    with st.spinner("Activating..."):
                        # We use a custom endpoint or the settings endpoint
                        response = api_client.post(f"/api/v1/licensing/activate?key={license_key}")
                        if response:
                            msg = response.get('message', 'License activated successfully!')
                            st.success(f"✅ {msg}")
                            st.info("Restarting application...")
                            st.session_state.license_valid = True
                            st.rerun()
                        else:
                            st.error("❌ Activation failed. Please check your license key.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")
    st.caption("Need help? Contact support at info@notin.ai")
