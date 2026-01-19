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
# ALIASES: Mapping of friendly names to file paths
# =============================================================================

NOTEBOOKS = {
    # Core notebooks in source/notebooks/
    "load_datasets": "source/notebooks/load_datasets.ipynb",
    "model_typology": "source/notebooks/model_typology.ipynb",
    "noise_ceilings": "source/notebooks/noise_ceilings.ipynb",
    "feature_stats": "source/notebooks/feature_stats.ipynb",
    "subject_regressions": "source/notebooks/subject_regressions.ipynb",
    "bash_scripting": "source/notebooks/bash_scripting.ipynb",
    # Historical notebooks in source/history/
    "effective_dims": "source/history/effective_dims.ipynb",
    "feature_metrics": "source/history/feature_metrics.ipynb",
    "processing1": "source/history/processing1.ipynb",
    "processing2": "source/history/processing2.ipynb",
}

SCRIPTS = {
    # Analysis scripts in source/scripts/
    "boot_regression": "source/scripts/boot_regression.py",
    "cross_decoding": "source/scripts/cross_decoding.py",
    "dataset_bootstrap": "source/scripts/dataset_bootstrap.py",
    "feature_analysis": "source/scripts/feature_analysis.py",
    "metric_permutations": "source/scripts/metric_permutations.py",
    "script_subject_regressions": "source/scripts/subject_regressions.py",
    # Historical scripts in source/history/
    "feature_regression": "source/history/feature_regression.py",
    "get_feature_maps": "source/history/get_feature_maps.py",
    "get_feature_metrics1": "source/history/get_feature_metrics1.py",
    "get_feature_metrics2": "source/history/get_feature_metrics2.py",
    "get_sparsity_maps": "source/history/get_sparsity_maps.py",
    "stepwise_regressions1": "source/history/stepwise_regressions1.py",
    "stepwise_regressions2": "source/history/stepwise_regressions2.py",
    "stepwise_regressions3": "source/history/stepwise_regressions3.py",
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

def execute_notebook(
    notebook_path: Path,
    output_dir: Optional[Path] = None,
    timeout: int = 600
) -> ExecutionResult:
    """
    Execute a Jupyter notebook using nbconvert.
    
    Args:
        notebook_path: Path to the notebook file
        output_dir: Directory to save executed notebook (optional)
        timeout: Execution timeout in seconds
    
    Returns:
        ExecutionResult with success status and captured output
    """
    target = notebook_path.stem
    start_time = time.time()
    
    # Build nbconvert command
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace" if output_dir is None else "",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook_path)
    ]
    
    # Remove empty strings from command
    cmd = [c for c in cmd if c]
    
    # Add output path if specified
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / notebook_path.name
        cmd.extend(["--output", str(output_path)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30  # Add buffer for nbconvert overhead
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        return ExecutionResult(
            target=target,
            path=str(notebook_path),
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

def execute_script(
    script_path: Path,
    args: Optional[list] = None,
    timeout: int = 600
) -> ExecutionResult:
    """
    Execute a Python script.
    
    Args:
        script_path: Path to the Python script
        args: Additional command-line arguments
        timeout: Execution timeout in seconds
    
    Returns:
        ExecutionResult with success status and captured output
    """
    target = script_path.stem
    start_time = time.time()
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_path.parent
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
    verbose: bool = False
) -> list:
    """
    Run multiple targets with a progress bar.
    
    Args:
        targets: List of (alias, path) tuples
        project_root: Project root directory
        output_dir: Output directory for notebooks
        timeout: Execution timeout per target
        verbose: Whether to print detailed output
    
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
    
    for alias, rel_path in iterator:
        path = project_root / rel_path
        
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
        
        # Update progress bar description
        if use_tqdm:
            iterator.set_description(f"Running {alias}")
        else:
            print_info(f"Running {alias}...")
        
        # Execute based on file type
        if path.suffix == ".ipynb":
            result = execute_notebook(path, output_dir, timeout)
        elif path.suffix == ".py":
            result = execute_script(path, timeout=timeout)
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
    
    print(f"\n{Colors.BOLD}Scripts:{Colors.ENDC}")
    for alias, path in sorted(SCRIPTS.items()):
        print(f"  {Colors.CYAN}{alias:30}{Colors.ENDC} -> {path}")
    
    print(f"\n{Colors.DIM}Use --all to run everything, or specify targets by name.{Colors.ENDC}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce analyses from AffectExMachina",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python reproduce.py --list              # List available targets
    python reproduce.py --all               # Run all analyses
    python reproduce.py load_datasets       # Run specific target
    python reproduce.py --notebooks         # Run all notebooks
    python reproduce.py --scripts           # Run all scripts
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
        help="Run all notebooks and scripts"
    )
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help="Run all notebooks only"
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Run all scripts only"
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
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_targets()
        return 0
    
    project_root = get_project_root()
    os.chdir(project_root)
    
    # Determine targets to run
    targets = []
    
    if args.all:
        targets = list(ALIASES.items())
    elif args.notebooks:
        targets = list(NOTEBOOKS.items())
    elif args.scripts:
        targets = list(SCRIPTS.items())
    elif args.targets:
        for t in args.targets:
            if t in ALIASES:
                targets.append((t, ALIASES[t]))
            else:
                # Check if it's a direct path
                path = Path(t)
                if path.exists():
                    targets.append((path.stem, str(path)))
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
        args.verbose
    )
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on results
    failed = [r for r in results if not r.success]
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
