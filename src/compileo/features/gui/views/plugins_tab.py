import streamlit as st
import pandas as pd
from src.compileo.features.gui.services.api_client import api_client

def render_plugins_tab():
    """Render the plugins management tab."""
    st.header("🧩 Plugins")
    st.markdown("Extend Compileo's functionality with plugins.")

    # Upload Section
    st.subheader("Install Plugin")
    uploaded_file = st.file_uploader("Upload Plugin (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("Install"):
            with st.spinner("Installing plugin..."):
                try:
                    # Use the API client to upload the file
                    # We assume api_client has a generic upload method or we add one
                    # Since api_client might not have a dedicated method, we'll use requests directly or extend it.
                    # For now, let's assume we can extend api_client or use st.session_state's api_base_url
                    
                    # Construct URL
                    base_url = api_client.base_url
                    url = f"{base_url}/api/v1/plugins/upload"
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/zip")}
                    headers = {}
                    if api_client.api_key:
                        headers["X-API-Key"] = str(api_client.api_key)
                    
                    import requests
                    response = requests.post(url, files=files, headers=headers)
                    
                    if response.status_code == 200:
                        st.success(f"Plugin installed successfully: {response.json().get('plugin_id')}")
                        st.rerun()
                    else:
                        st.error(f"Failed to install plugin: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error installing plugin: {e}")

    # List Plugins
    st.subheader("Installed Plugins")
    
    try:
        # Fetch plugins list
        base_url = api_client.base_url
        url = f"{base_url}/api/v1/plugins/"
        headers = {}
        if api_client.api_key:
            headers["X-API-Key"] = str(api_client.api_key)
        
        import requests
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            plugins = response.json()
            if not plugins:
                st.info("No plugins installed.")
            else:
                # Display as a dataframe or list
                for plugin in plugins:
                    with st.expander(f"{plugin['name']} (v{plugin['version']})"):
                        st.write(f"**ID:** {plugin['id']}")
                        st.write(f"**Author:** {plugin['author']}")
                        st.write(f"**Description:** {plugin['description']}")
                        st.write(f"**Entry Point:** {plugin['entry_point']}")
                        
                        # Uninstall Button
                        if st.button("🗑️ Uninstall", key=f"uninstall_{plugin['id']}"):
                            with st.spinner(f"Uninstalling {plugin['name']}..."):
                                try:
                                    del_url = f"{base_url}/api/v1/plugins/{plugin['id']}"
                                    del_resp = requests.delete(del_url, headers=headers)
                                    if del_resp.status_code == 200:
                                        st.success(f"Uninstalled {plugin['name']}")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to uninstall: {del_resp.text}")
                                except Exception as e:
                                    st.error(f"Error uninstalling: {e}")
        else:
             st.error(f"Failed to fetch plugins: {response.text}")

    except Exception as e:
        st.error(f"Error loading plugins: {e}")