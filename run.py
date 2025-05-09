import subprocess
import pkg_resources
import os
import sys

def check_and_install_requirements():
    """Check and install dependencies from requirements.txt."""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        to_install = []
        
        for req in requirements:
            try:
                pkg_resources.require(req)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                to_install.append(req)
        
        if to_install:
            print(f"Installing missing dependencies: {', '.join(to_install)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *to_install])
            print("Dependencies installed successfully!")
        else:
            print("All dependencies are already installed.")
    
    except FileNotFoundError:
        print("Error: requirements.txt not found in the current directory.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def check_chroma_db():
    """Verify chroma_db directory exists."""
    if not os.path.isdir('chroma_db'):
        print("Error: chroma_db directory not found in the current directory.")
        sys.exit(1)
    print("chroma_db directory found.")

def run_streamlit():
    """Run the Streamlit app."""
    try:
        print("Starting Smart Career Advisor...")
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.fileWatcherType', 'none'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: streamlit_app.py not found in the current directory.")
        sys.exit(1)

def main():
    print("Setting up Smart Career Advisor...")
    check_and_install_requirements()
    check_chroma_db()
    run_streamlit()

if __name__ == "__main__":
    main()