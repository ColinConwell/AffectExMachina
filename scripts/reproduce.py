#!/usr/bin/env python3
"""
Reproduction script for AffectExMachina analyses.

Run Jupyter notebooks and Python scripts from source/ to reproduce
results from the paper.

Usage:
    python reproduce.py --list              # List available targets
    python reproduce.py --all               # Run all analyses
    python reproduce.py load_datasets       # Run specific target by alias
    python reproduce.py --notebooks         # Run all notebooks
    python reproduce.py --scripts           # Run all scripts
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# ALIASES: Mapping of analysis names / targets to file paths
# =============================================================================

NOTEBOOKS = {
    # Core notebooks in source/notebooks/
    "load_datasets": "source/notebooks/load_datasets.ipynb",
    "model_typology": "source/notebooks/model_typology.ipynb",
    "noise_ceilings": "source/notebooks/noise_ceilings.ipynb",
    "feature_stats": "source/notebooks/feature_stats.ipynb",
    "subject_regressions": "source/notebooks/subject_regressions.ipynb",
    "bash_scripting": "source/notebooks/bash_scripting.ipynb",
}

# Scripts use relative imports and run as modules from project root.
# They typically require command-line arguments (e.g., --model_string).
SCRIPTS = {
    # Analysis scripts in source/scripts/ (run as modules from project root)
    "regression_bootstrap": "source.scripts.boot_regression",
    "cross_decoding": "source.scripts.cross_decoding",
    "dataset_bootstrap": "source.scripts.dataset_bootstrap",
    "feature_analysis": "source.scripts.feature_analysis",
    "metric_permutations": "source.scripts.metric_permutations",
    "script_subject_regressions": "source.scripts.subject_regressions",
}

# Combined aliases
ALIASES = {**NOTEBOOKS, **SCRIPTS}

# =============================================================================
# ANSI Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing a notebook or script."""
    target: str
    path: str
    success: bool
    duration: float
    stdout: str
    stderr: str
    error_message: Optional[str] = None

# =============================================================================
# Helper functions
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()

def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

def print_success(text: str) -> None:
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")

def print_error(text: str) -> None:
    print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {text}")

def print_info(text: str) -> None:
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")

def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"

# =============================================================================
# Execution functions
# =============================================================================

def get_source_dir() -> Path:
    """Get the source directory."""
    return get_project_root() / "source"

def execute_notebook(
    notebook_path: Path,
    output_dir: Optional[Path] = None,
    timeout: int = 600,
    kernel_name: Optional[str] = None
) -> ExecutionResult:
    """
    Execute a Jupyter notebook using papermill.
    
    Papermill properly sets the kernel's working directory, allowing notebooks
    to use relative paths for data files.
    
    Args:
        notebook_path: Path to the notebook file
        output_dir: Directory to save executed notebook (optional)
        timeout: Execution timeout in seconds
        kernel_name: Kernel name to use (defaults to 'python3')
    
    Returns:
        ExecutionResult with success status and captured output
    """
    target = notebook_path.stem
    start_time = time.time()
    
    source_dir = get_source_dir()
    
    # Convert notebook path to absolute
    notebook_path = notebook_path.resolve()
    notebook_dir = notebook_path.parent
    
    # Determine output path
    # By default, write to a temp file to avoid modifying the original notebook
    # (papermill adds error markers and execution metadata to notebooks)
    import tempfile
    temp_output = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / notebook_path.name
    else:
        # Use temp file to avoid modifying original notebook
        temp_output = tempfile.NamedTemporaryFile(
            suffix='.ipynb', delete=False, dir=notebook_dir
        )
        output_path = Path(temp_output.name)
        temp_output.close()
    
    # Build papermill command
    kernel = kernel_name or "python3"
    cmd = [
        sys.executable, "-m", "papermill",
        str(notebook_path),
        str(output_path),
        "--kernel", kernel,
        "--cwd", str(source_dir),  # Set kernel working directory
    ]
    
    # Set up environment:
    # - notebook_dir in PYTHONPATH for local __init__.py imports
    # - source_dir in PYTHONPATH for package imports
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    paths = [str(notebook_dir), str(source_dir)]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = ":".join(paths)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=str(source_dir),
            env=env
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        # Check for execution errors in the output
        error_indicators = ["PapermillExecutionError", "CellExecutionError", "Exception"]
        if success:
            for indicator in error_indicators:
                if indicator in result.stderr or indicator in result.stdout:
                    success = False
                    break
        
        error_msg = None
        if not success:
            # Extract relevant error message
            error_msg = result.stderr if result.stderr else result.stdout
            # Truncate if too long
            if len(error_msg) > 1000:
                error_msg = error_msg[:500] + "\n...\n" + error_msg[-500:]
        
        return ExecutionResult(
            target=target,
            path=str(notebook_path),
            success=success,
            duration=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=error_msg
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return ExecutionResult(
            target=target,
            path=str(notebook_path),
            success=False,
            duration=duration,
            stdout="",
            stderr="",
            error_message=f"Execution timed out after {timeout}s"
        )
    except Exception as e:
        duration = time.time() - start_time
        return ExecutionResult(
            target=target,
            path=str(notebook_path),
            success=False,
            duration=duration,
            stdout="",
            stderr="",
            error_message=str(e)
        )
    finally:
        # Clean up temp output file if used
        if temp_output is not None and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

def execute_module(
    module_name: str,
    args: Optional[list] = None,
    timeout: int = 600
) -> ExecutionResult:
    """
    Execute a Python module using python -m.
    
    Modules are run from project root with source/ in PYTHONPATH.
    
    Args:
        module_name: Module name (e.g., 'source.scripts.boot_regression')
        args: Additional command-line arguments
        timeout: Execution timeout in seconds
    
    Returns:
        ExecutionResult with success status and captured output
    """
    target = module_name.split(".")[-1]
    start_time = time.time()
    
    project_root = get_project_root()
    source_dir = get_source_dir()
    
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    
    # Set up environment with source/ in PYTHONPATH for absolute imports
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    paths = [str(source_dir)]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = ":".join(paths)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),  # Run from project root
            env=env
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        error_msg = None
        if not success:
            error_msg = result.stderr if result.stderr else result.stdout
            if len(error_msg) > 1000:
                error_msg = error_msg[:500] + "\n...\n" + error_msg[-500:]
        
        return ExecutionResult(
            target=target,
            path=module_name,
            success=success,
            duration=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=error_msg
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return ExecutionResult(
            target=target,
            path=module_name,
            success=False,
            duration=duration,
            stdout="",
            stderr="",
            error_message=f"Execution timed out after {timeout}s"
        )
    except Exception as e:
        duration = time.time() - start_time
        return ExecutionResult(
            target=target,
            path=module_name,
            success=False,
            duration=duration,
            stdout="",
            stderr="",
            error_message=str(e)
        )

def execute_script(
    script_path: Path,
    args: Optional[list] = None,
    timeout: int = 600
) -> ExecutionResult:
    """
    Execute a Python script directly.
    
    Args:
        script_path: Path to the Python script
        args: Additional command-line arguments
        timeout: Execution timeout in seconds
    
    Returns:
        ExecutionResult with success status and captured output
    """
    target = script_path.stem
    start_time = time.time()
    
    # Convert to absolute path
    script_path = script_path.resolve()
    
    # Run from script's parent directory
    script_dir = script_path.parent
    source_dir = get_source_dir()
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    # Set up environment with source/ in PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    paths = [str(script_dir), str(source_dir)]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = ":".join(paths)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_dir),
            env=env
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        return ExecutionResult(
            target=target,
            path=str(script_path),
            success=success,
            duration=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=result.stderr if not success else None
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return ExecutionResult(
            target=target,
            path=str(script_path),
            success=False,
            duration=duration,
            stdout="",
            stderr="",
            error_message=f"Execution timed out after {timeout}s"
        )
    except Exception as e:
        duration = time.time() - start_time
        return ExecutionResult(
            target=target,
            path=str(script_path),
            success=False,
            duration=duration,
            stdout="",
            stderr="",
            error_message=str(e)
        )

def run_targets(
    targets: list,
    project_root: Path,
    output_dir: Optional[Path] = None,
    timeout: int = 600,
    verbose: bool = False,
    extra_args: Optional[list] = None
) -> list:
    """
    Run multiple targets with a progress bar.
    
    Args:
        targets: List of (alias, path_or_module, is_module) tuples
        project_root: Project root directory
        output_dir: Output directory for notebooks
        timeout: Execution timeout per target
        verbose: Whether to print detailed output
        extra_args: Additional arguments to pass to scripts/notebooks
    
    Returns:
        List of ExecutionResult objects
    """
    results = []
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = len(targets) > 1
    except ImportError:
        use_tqdm = False
        print_info("Install tqdm for progress bars: pip install tqdm")
    
    iterator = tqdm(targets, desc="Running analyses", unit="target") if use_tqdm else targets
    
    for item in iterator:
        # Unpack target info
        if len(item) == 3:
            alias, path_or_module, is_module = item
        else:
            alias, path_or_module = item
            is_module = False
        
        # Update progress bar description
        if use_tqdm:
            iterator.set_description(f"Running {alias}")
        else:
            print_info(f"Running {alias}...")
        
        # Execute based on type
        if is_module:
            # Run as Python module
            result = execute_module(path_or_module, args=extra_args, timeout=timeout)
        else:
            path = project_root / path_or_module
            
            if not path.exists():
                results.append(ExecutionResult(
                    target=alias,
                    path=str(path),
                    success=False,
                    duration=0,
                    stdout="",
                    stderr="",
                    error_message=f"File not found: {path}"
                ))
                continue
            
            if path.suffix == ".ipynb":
                result = execute_notebook(path, output_dir, timeout)
            elif path.suffix == ".py":
                result = execute_script(path, args=extra_args, timeout=timeout)
            else:
                results.append(ExecutionResult(
                    target=alias,
                    path=str(path),
                    success=False,
                    duration=0,
                    stdout="",
                    stderr="",
                    error_message=f"Unknown file type: {path.suffix}"
                ))
                continue
        
        results.append(result)
        
        # Print result if not using tqdm or if verbose
        if not use_tqdm or verbose:
            if result.success:
                print_success(f"{alias} completed in {format_duration(result.duration)}")
            else:
                print_error(f"{alias} failed: {result.error_message}")
    
    return results

def print_summary(results: list) -> None:
    """Print a summary of execution results."""
    print_header("Execution Summary")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    total_duration = sum(r.duration for r in results)
    
    print(f"Total targets: {len(results)}")
    print(f"Successful: {Colors.GREEN}{len(successful)}{Colors.ENDC}")
    print(f"Failed: {Colors.FAIL}{len(failed)}{Colors.ENDC}")
    print(f"Total time: {format_duration(total_duration)}")
    
    if successful:
        print(f"\n{Colors.GREEN}Successful:{Colors.ENDC}")
        for r in successful:
            print(f"  - {r.target} ({format_duration(r.duration)})")
    
    if failed:
        print(f"\n{Colors.FAIL}Failed:{Colors.ENDC}")
        for r in failed:
            print(f"  - {r.target}: {r.error_message}")

def list_targets() -> None:
    """Print available targets."""
    print_header("Available Targets")
    
    print(f"{Colors.BOLD}Notebooks:{Colors.ENDC}")
    for alias, path in sorted(NOTEBOOKS.items()):
        print(f"  {Colors.CYAN}{alias:30}{Colors.ENDC} -> {path}")
    
    if SCRIPTS:
        print(f"\n{Colors.BOLD}Scripts (run as modules, may require arguments):{Colors.ENDC}")
        for alias, module in sorted(SCRIPTS.items()):
            print(f"  {Colors.CYAN}{alias:30}{Colors.ENDC} -> python -m {module}")
        print(f"\n{Colors.DIM}  Pass arguments after '--': reproduce.py <script> -- --arg1 value1{Colors.ENDC}")
    
    print(f"\n{Colors.DIM}Use --notebooks to run all notebooks, or specify targets by name.{Colors.ENDC}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce analyses from AffectExMachina",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python reproduce.py --list                        # List available targets
    python reproduce.py load_datasets                 # Run specific notebook
    python reproduce.py --notebooks                   # Run all notebooks
    python reproduce.py regression_bootstrap -- --help  # Run script with args
    python reproduce.py feature_analysis -- --model_string resnet50 --imageset oasis
        """
    )
    
    parser.add_argument(
        "targets",
        nargs="*",
        help="Target names or aliases to run"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all notebooks (scripts require arguments)"
    )
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help="Run all notebooks only"
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Run all scripts (will likely fail without arguments)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available targets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save executed notebooks"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Execution timeout per target in seconds (default: 600)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    
    # Handle -- separator manually for passing args to scripts
    argv = sys.argv[1:]
    extra_args = []
    if "--" in argv:
        sep_idx = argv.index("--")
        extra_args = argv[sep_idx + 1:]
        argv = argv[:sep_idx]
    
    args = parser.parse_args(argv)
    
    # Handle --list
    if args.list:
        list_targets()
        return 0
    
    project_root = get_project_root()
    os.chdir(project_root)
    
    # Determine targets to run
    # Format: (alias, path_or_module, is_module)
    targets = []
    
    if args.all:
        # Run notebooks only by default (scripts need args)
        for alias, path in NOTEBOOKS.items():
            targets.append((alias, path, False))
    elif args.notebooks:
        for alias, path in NOTEBOOKS.items():
            targets.append((alias, path, False))
    elif args.scripts:
        for alias, module in SCRIPTS.items():
            targets.append((alias, module, True))
    elif args.targets:
        for t in args.targets:
            if t in NOTEBOOKS:
                targets.append((t, NOTEBOOKS[t], False))
            elif t in SCRIPTS:
                targets.append((t, SCRIPTS[t], True))
            else:
                # Check if it's a direct path
                path = Path(t)
                if path.exists():
                    targets.append((path.stem, str(path), False))
                else:
                    print_error(f"Unknown target: {t}")
                    print_info("Use --list to see available targets")
                    return 1
    else:
        parser.print_help()
        return 0
    
    if not targets:
        print_error("No targets to run")
        return 1
    
    print_header("AffectExMachina Reproduction")
    print_info(f"Project root: {project_root}")
    print_info(f"Targets to run: {len(targets)}")
    if extra_args:
        print_info(f"Extra arguments: {' '.join(extra_args)}")
    
    # Set up output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Output directory: {output_dir}")
    
    # Run targets
    results = run_targets(
        targets,
        project_root,
        output_dir,
        args.timeout,
        args.verbose,
        extra_args if extra_args else None
    )
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on results
    failed = [r for r in results if not r.success]
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
