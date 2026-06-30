import sys
import os
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Point Streamlit to run the main streamlit_app.py script
    script_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())
