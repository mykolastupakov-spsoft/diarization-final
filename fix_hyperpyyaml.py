#!/usr/bin/env python3
"""Fix hyperpyyaml for Python 3.14 compatibility"""

import os
import shutil

hyperpyyaml_file = "/opt/homebrew/lib/python3.14/site-packages/hyperpyyaml/core.py"

if not os.path.exists(hyperpyyaml_file):
    print(f"File not found: {hyperpyyaml_file}")
    exit(1)

# Read the file
with open(hyperpyyaml_file, 'r') as f:
    content = f.read()

# Check if already patched
if "ast.Constant" in content and "isinstance(node.value, (int, float, complex))" in content:
    print("Already patched!")
    exit(0)

# Create backup
backup_file = hyperpyyaml_file + ".backup"
if not os.path.exists(backup_file):
    shutil.copy2(hyperpyyaml_file, backup_file)
    print(f"Created backup: {backup_file}")

# Replace ast.Num with ast.Constant
old_code = """    if isinstance(node, ast.Num):  # <number>
        return node.n"""
new_code = """    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):  # <number> (Python 3.8+)
        return node.value"""

if old_code in content:
    content = content.replace(old_code, new_code)
    
    # Write back
    try:
        with open(hyperpyyaml_file, 'w') as f:
            f.write(content)
        print("✅ Successfully patched hyperpyyaml!")
    except PermissionError:
        print("❌ Permission denied. Please run with sudo:")
        print(f"   sudo python3 {__file__}")
        exit(1)
else:
    print("❌ Could not find the code to patch")
    exit(1)

