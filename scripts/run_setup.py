#!/usr/bin/env python3
"""
Interactive setup script for AffectExMachina repository.

Guides users through environment setup with options for:
- Creating virtual environments (uv, conda, or venv)
- Installing dependencies (base and optional groups)
- Verifying installation
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")

def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {text}")

def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {text}")

def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no response."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{Colors.BLUE}[?]{Colors.ENDC} {question} [{default_str}]: ").strip().lower()
        if response == "":
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")

def prompt_choice(question: str, choices: list, default: int = 0) -> str:
    """Prompt user to select from a list of choices."""
    print(f"\n{Colors.BLUE}[?]{Colors.ENDC} {question}")
    for i, choice in enumerate(choices):
        marker = " (default)" if i == default else ""
        print(f"    {i + 1}. {choice}{marker}")
    
    while True:
        response = input(f"    Enter choice [1-{len(choices)}]: ").strip()
        if response == "":
            return choices[default]
        try:
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print(f"    Please enter a number between 1 and {len(choices)}.")

def run_command(cmd: list, capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(f"    {e.stderr}")
        raise

def check_python_version() -> bool:
    """Check if Python version is compatible (>=3.9)."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 9:
        print_success(f"Python {version_str} detected (>= 3.9 required)")
        return True
    else:
        print_error(f"Python {version_str} detected, but >= 3.9 is required")
        return False

def detect_package_managers() -> dict:
    """Detect available package managers."""
    managers = {}
    
    # Check for uv
    if shutil.which("uv"):
        managers["uv"] = shutil.which("uv")
    
    # Check for conda/mamba
    if shutil.which("conda"):
        managers["conda"] = shutil.which("conda")
    if shutil.which("mamba"):
        managers["mamba"] = shutil.which("mamba")
    
    # pip is typically available with Python
    if shutil.which("pip") or shutil.which("pip3"):
        managers["pip"] = shutil.which("pip") or shutil.which("pip3")
    
    return managers

def get_project_root() -> Path:
    """Get the project root directory."""
    # This script is in scripts/, so parent is project root
    return Path(__file__).parent.parent.resolve()

def create_uv_environment(project_root: Path) -> bool:
    """Create a virtual environment using uv."""
    print_info("Creating virtual environment with uv...")
    
    venv_path = project_root / ".venv"
    if venv_path.exists():
        if not prompt_yes_no(f"Virtual environment already exists at {venv_path}. Recreate?", default=False):
            print_info("Using existing virtual environment.")
            return True
    
    try:
        run_command(["uv", "venv", str(venv_path)])
        print_success(f"Virtual environment created at {venv_path}")
        print_info(f"Activate with: source {venv_path}/bin/activate")
        return True
    except subprocess.CalledProcessError:
        return False

def create_conda_environment(project_root: Path, manager: str = "conda") -> bool:
    """Create a conda environment."""
    env_name = input(f"{Colors.BLUE}[?]{Colors.ENDC} Enter environment name [affectexmachina]: ").strip()
    if not env_name:
        env_name = "affectexmachina"
    
    print_info(f"Creating conda environment '{env_name}'...")
    
    try:
        run_command([manager, "create", "-n", env_name, "python=3.10", "-y"])
        print_success(f"Conda environment '{env_name}' created")
        print_info(f"Activate with: conda activate {env_name}")
        return True
    except subprocess.CalledProcessError:
        return False

def create_venv_environment(project_root: Path) -> bool:
    """Create a virtual environment using built-in venv."""
    print_info("Creating virtual environment with venv...")
    
    venv_path = project_root / ".venv"
    if venv_path.exists():
        if not prompt_yes_no(f"Virtual environment already exists at {venv_path}. Recreate?", default=False):
            print_info("Using existing virtual environment.")
            return True
    
    try:
        run_command([sys.executable, "-m", "venv", str(venv_path)])
        print_success(f"Virtual environment created at {venv_path}")
        print_info(f"Activate with: source {venv_path}/bin/activate")
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies_uv(project_root: Path, extras: list) -> bool:
    """Install dependencies using uv."""
    print_info("Installing dependencies with uv...")
    
    extras_str = ",".join(extras) if extras else ""
    install_spec = f"-e .[{extras_str}]" if extras_str else "-e ."
    
    try:
        run_command(["uv", "pip", "install", install_spec], check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies_pip(project_root: Path, extras: list) -> bool:
    """Install dependencies using pip."""
    print_info("Installing dependencies with pip...")
    
    extras_str = ",".join(extras) if extras else ""
    install_spec = f".[{extras_str}]" if extras_str else "."
    
    try:
        run_command([sys.executable, "-m", "pip", "install", "-e", install_spec], check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies_conda(project_root: Path, manager: str = "conda") -> bool:
    """Install dependencies using conda (base packages) then pip (extras)."""
    print_info(f"Installing base dependencies with {manager}...")
    
    # Install core packages via conda
    core_packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "tqdm", "pillow"]
    
    try:
        run_command([manager, "install", "-y"] + core_packages, check=True)
        print_success("Core dependencies installed via conda")
        
        # Install remaining dependencies via pip
        print_info("Installing additional dependencies with pip...")
        run_command([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print_success("Additional dependencies installed")
        return True
    except subprocess.CalledProcessError:
        return False

def verify_installation() -> bool:
    """Verify that key packages are importable."""
    print_header("Verifying Installation")
    
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("tqdm", "tqdm"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    optional_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "timm"),
        ("plotnine", "plotnine"),
    ]
    
    all_ok = True
    
    print_info("Checking core packages...")
    for module, name in packages:
        try:
            __import__(module)
            print_success(f"{name} imported successfully")
        except ImportError:
            print_error(f"{name} could not be imported")
            all_ok = False
    
    print_info("\nChecking optional packages...")
    for module, name in optional_packages:
        try:
            __import__(module)
            print_success(f"{name} imported successfully")
        except ImportError:
            print_warning(f"{name} not installed (optional)")
    
    return all_ok

def main():
    """Main setup workflow."""
    print_header("AffectExMachina Setup")
    print("This script will guide you through setting up the development environment.\n")
    
    project_root = get_project_root()
    print_info(f"Project root: {project_root}")
    
    # Step 1: Check Python version
    if not check_python_version():
        print_error("Please install Python 3.9 or higher and try again.")
        sys.exit(1)
    
    # Step 2: Detect package managers
    print_header("Detecting Package Managers")
    managers = detect_package_managers()
    
    for name, path in managers.items():
        print_success(f"{name} found at {path}")
    
    if not managers:
        print_error("No package managers found. Please install pip, uv, or conda.")
        sys.exit(1)
    
    # Step 3: Create virtual environment
    print_header("Virtual Environment Setup")
    
    if prompt_yes_no("Would you like to create a virtual environment?", default=True):
        env_choices = []
        if "uv" in managers:
            env_choices.append("uv (recommended)")
        if "conda" in managers or "mamba" in managers:
            env_choices.append("conda/mamba")
        env_choices.append("venv (built-in)")
        env_choices.append("Skip")
        
        env_choice = prompt_choice("Select environment manager:", env_choices)
        
        os.chdir(project_root)
        
        if "uv" in env_choice:
            create_uv_environment(project_root)
        elif "conda" in env_choice:
            manager = "mamba" if "mamba" in managers else "conda"
            create_conda_environment(project_root, manager)
        elif "venv" in env_choice:
            create_venv_environment(project_root)
        else:
            print_info("Skipping virtual environment creation.")
    
    # Step 4: Install dependencies
    print_header("Dependency Installation")
    
    if prompt_yes_no("Would you like to install dependencies?", default=True):
        # Ask about optional dependency groups
        print_info("\nAvailable optional dependency groups:")
        print("  - pytorch: PyTorch, TorchVision, timm (for model inference)")
        print("  - stats: numba, pingouin, ftfy, regex (for statistical analysis)")
        print("  - viz: plotnine, siuba (for visualization)")
        print("  - external: visualpriors, CLIP (external model repos)")
        
        extras = []
        if prompt_yes_no("Install PyTorch dependencies?", default=True):
            extras.append("pytorch")
        if prompt_yes_no("Install statistics dependencies?", default=True):
            extras.append("stats")
        if prompt_yes_no("Install visualization dependencies?", default=True):
            extras.append("viz")
        if prompt_yes_no("Install external model dependencies?", default=False):
            extras.append("external")
        
        # Choose installation method
        install_choices = []
        if "uv" in managers:
            install_choices.append("uv (recommended)")
        if "pip" in managers:
            install_choices.append("pip")
        if "conda" in managers or "mamba" in managers:
            install_choices.append("conda (then pip for extras)")
        
        install_choice = prompt_choice("Select installation method:", install_choices)
        
        os.chdir(project_root)
        
        if "uv" in install_choice:
            install_dependencies_uv(project_root, extras)
        elif "pip" in install_choice:
            install_dependencies_pip(project_root, extras)
        elif "conda" in install_choice:
            manager = "mamba" if "mamba" in managers else "conda"
            install_dependencies_conda(project_root, manager)
    
    # Step 5: Verify installation
    if prompt_yes_no("Would you like to verify the installation?", default=True):
        verify_installation()
    
    # Done
    print_header("Setup Complete")
    print_success("Environment setup is complete!")
    print_info("\nNext steps:")
    print("  1. Activate your virtual environment (if created)")
    print("  2. Run 'python scripts/reproduce.py --list' to see available analyses")
    print("  3. Run 'python scripts/reproduce.py --all' to reproduce all results")
    print("  4. See guidebook/overview.md for detailed documentation")

if __name__ == "__main__":
    main()
