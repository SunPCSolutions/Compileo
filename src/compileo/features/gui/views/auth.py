"""
Authentication views for Compileo GUI.
"""

import streamlit as st
from src.compileo.features.gui.services.api_client import api_client
from src.compileo.features.gui.state.session_state import session_state

def render_login(cookie_manager=None):
    """Render login page."""
    st.markdown('<div class="main-header">🔒 Login to Compileo</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            username = st.text_input("Email or Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    with st.spinner("Authenticating..."):
                        # FastAPI OAuth2 expects form data
                        login_data = {
                            "username": username,
                            "password": password
                        }
                        response = api_client.post("/auth/login", data=login_data, use_json=False)
                        
                        if response and "access_token" in response:
                            token = response["access_token"]
                            session_state.jwt_token = token
                            api_client.set_jwt_token(token)
                            
                            # Fetch user profile to populate session_state.user
                            try:
                                user_info = api_client.get("/auth/me")
                                if user_info:
                                    session_state.user = user_info
                                else:
                                    # Fallback if /auth/me fails but login succeeded
                                    session_state.user = {"id": "default_user", "email": username}
                            except Exception:
                                session_state.user = {"id": "default_user", "email": username}
                                
                            session_state.authenticated = True
                            
                            # Clear logout flag to allow future auto-logins
                            if 'logging_out' in st.session_state:
                                del st.session_state['logging_out']
                                
                            # Save to cookie for persistence
                            if cookie_manager:
                                cookie_manager.set("compileo_jwt_token", token)
                                # No need to set expires_at here as it defaults to reasonable value
                            
                            st.success("Successfully logged in!")
                            # Force a brief wait to ensure cookie is set before rerun
                            import time
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Login failed. Please check your credentials.")
            
            st.markdown("---")
            st.markdown("Don't have an account?")
            if st.button("Register New Account", use_container_width=True):
                st.session_state.auth_mode = "register"
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

def render_register(cookie_manager=None):
    """Render registration page."""
    st.markdown('<div class="main-header">📝 Create Compileo Account</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            email = st.text_input("Email")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Register", use_container_width=True, type="primary"):
                if not email or not username or not password:
                    st.error("Please fill in all required fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Creating account..."):
                        reg_data = {
                            "email": email,
                            "username": username,
                            "password": password
                        }
                        response = api_client.post("/auth/register", data=reg_data)
                        
                        if response and "id" in response:
                            st.success("Account created successfully! Please login.")
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error("Registration failed. Email or username might already be taken.")
            
            st.markdown("---")
            if st.button("Back to Login", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

def render_auth_page(cookie_manager=None):
    """Main authentication entry point."""
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = "login"
        
    if st.session_state.auth_mode == "register":
        render_register(cookie_manager=cookie_manager)
    else:
        render_login(cookie_manager=cookie_manager)
