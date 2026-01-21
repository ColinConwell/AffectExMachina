#!/usr/bin/env python3
"""
Clean BibTeX files by removing entries not referenced in source documents.

Usage:
    python clean_bibtex.py SOURCE_DOC BIBTEX_FILE [-o OUTPUT_FILE]
    
Examples:
    python clean_bibtex.py overview.qmd references.bib
"""

import re, argparse, shutil
from pathlib import Path
from typing import Set, Dict, Tuple


def extract_citation_keys_from_document(source_path: Path) -> Set[str]:
    """
    Extract all citation keys from a source document (.tex, .qmd, etc.).
    
    Supports common citation commands:
        - LaTeX: \\cite{}, \\citep{}, \\citet{}, \\citeauthor{}, \\citeyear{}, etc.
        - Pandoc/Quarto: @key, [@key], [@key1; @key2]
    
    Args:
        source_path: Path to the source document
        
    Returns:
        Set of citation keys found in the document
    """
    content = source_path.read_text(encoding='utf-8')
    keys: Set[str] = set()
    
    # LaTeX citation patterns: \cite{key}, \citep{key1,key2}, \citet{key}, etc.
    latex_pattern = r'\\cite[a-z]*\*?\{([^}]+)\}'
    for match in re.finditer(latex_pattern, content):
        # Split by comma to handle multiple keys in one citation
        citation_group = match.group(1)
        for key in citation_group.split(','):
            cleaned_key = key.strip()
            if cleaned_key:
                keys.add(cleaned_key)
    
    # Pandoc/Quarto citation patterns: @key, [@key], [@key1; @key2]
    # Match @key but not email addresses (preceded by alphanumeric)
    pandoc_pattern = r'(?<![a-zA-Z0-9])@([a-zA-Z][a-zA-Z0-9_:-]*)'
    for match in re.finditer(pandoc_pattern, content):
        keys.add(match.group(1))
    
    return keys


def parse_bibtex_entries(bibtex_path: Path) -> Dict[str, str]:
    """
    Parse a BibTeX file and extract all entries with their full text.
    
    Args:
        bibtex_path: Path to the BibTeX file
        
    Returns:
        Dictionary mapping citation keys to their full BibTeX entry text
    """
    content = bibtex_path.read_text(encoding='utf-8')
    entries: Dict[str, str] = {}
    
    # Pattern to match bibtex entries: @type{key, ... }
    # This handles nested braces properly
    entry_pattern = r'@(\w+)\s*\{\s*([^,\s]+)\s*,'
    
    # Find all entry starts
    for match in re.finditer(entry_pattern, content):
        entry_type = match.group(1).lower()
        key = match.group(2)
        start_pos = match.start()
        
        # Skip comments
        if entry_type == 'comment':
            continue
        
        # Find the matching closing brace by counting braces
        brace_count = 0
        end_pos = start_pos
        in_entry = False
        
        for i, char in enumerate(content[start_pos:], start=start_pos):
            if char == '{':
                brace_count += 1
                in_entry = True
            elif char == '}':
                brace_count -= 1
                if in_entry and brace_count == 0:
                    end_pos = i + 1
                    break
        
        entry_text = content[start_pos:end_pos]
        entries[key] = entry_text
    
    return entries


def filter_bibtex_entries(
    entries: Dict[str, str], 
    used_keys: Set[str]
) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """
    Filter BibTeX entries to keep only those referenced in the document.
    
    Args:
        entries: Dictionary of all BibTeX entries
        used_keys: Set of citation keys used in the source document
        
    Returns:
        Tuple of (filtered entries dict, kept keys, removed keys)
    """
    filtered = {}
    kept_keys: Set[str] = set()
    removed_keys: Set[str] = set()
    
    for key, entry in entries.items():
        if key in used_keys:
            filtered[key] = entry
            kept_keys.add(key)
        else:
            removed_keys.add(key)
    
    return filtered, kept_keys, removed_keys


def write_bibtex_file(entries: Dict[str, str], output_path: Path) -> None:
    """
    Write filtered BibTeX entries to a file.
    
    Args:
        entries: Dictionary of BibTeX entries to write
        output_path: Path to the output file
    """
    # Sort entries alphabetically by key for consistency
    sorted_entries = [entries[key] for key in sorted(entries.keys())]
    content = '\n\n'.join(sorted_entries)
    
    output_path.write_text(content + '\n', encoding='utf-8')


def clean_bibtex(
    source_path: Path,
    bibtex_path: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Main function to clean a BibTeX file based on references in a source document.
    
    Args:
        source_path: Path to the source document
        bibtex_path: Path to the BibTeX file
        output_path: Path for output (defaults to overwriting input)
        overwrite: If True, skip backup when overwriting input file
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (number of entries kept, number of entries removed)
    """
    if output_path is None:
        output_path = bibtex_path
    
    # Create backup if overwriting in place and --overwrite not specified
    if output_path.resolve() == bibtex_path.resolve() and not overwrite:
        backup_path = bibtex_path.with_stem(f"{bibtex_path.stem}_original")
        shutil.copy2(bibtex_path, backup_path)
        if verbose:
            print(f"Backup saved to: {backup_path}")
    
    # Extract citation keys from source
    used_keys = extract_citation_keys_from_document(source_path)
    if verbose:
        print(f"Found {len(used_keys)} citation keys in {source_path.name}")
    
    # Parse BibTeX entries
    entries = parse_bibtex_entries(bibtex_path)
    if verbose:
        print(f"Found {len(entries)} entries in {bibtex_path.name}")
    
    # Filter entries
    filtered, kept, removed = filter_bibtex_entries(entries, used_keys)
    
    # Check for missing references
    missing = used_keys - kept
    if missing and verbose:
        print(f"Warning: {len(missing)} cited keys not found in BibTeX file:")
        for key in sorted(missing)[:10]:
            print(f"  - {key}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    # Write output
    write_bibtex_file(filtered, output_path)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Kept: {len(kept)} entries")
        print(f"  Removed: {len(removed)} entries")
        print(f"  Output written to: {output_path}")
    
    return len(kept), len(removed)


def main():
    parser = argparse.ArgumentParser(
        description='Clean BibTeX file by removing unreferenced entries.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'source',
        type=Path,
        help='Source document (.tex, .qmd, etc.) containing citations'
    )
    parser.add_argument(
        'bibtex',
        type=Path,
        help='BibTeX file to clean'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output file (default: overwrite input)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output messages'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite input file without creating a backup'
    )
    
    args = parser.parse_args()
    
    if not args.source.exists():
        print(f"Error: Source file not found: {args.source}")
        return 1
    
    if not args.bibtex.exists():
        print(f"Error: BibTeX file not found: {args.bibtex}")
        return 1
    
    clean_bibtex(
        source_path=args.source,
        bibtex_path=args.bibtex,
        output_path=args.output,
        overwrite=args.overwrite,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
