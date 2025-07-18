#!/usr/bin/env python3
"""Test script to verify the server syntax is correct."""

import ast
import sys
import os

def check_syntax(file_path):
    """Check if the Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the file
        ast.parse(source)
        print(f"✓ {file_path} has valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path} has syntax error: {e}")
        print(f"  Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"✗ {file_path} error: {e}")
        return False

def main():
    """Test all Python files in the project."""
    print("Testing Playwright MCP Server Syntax...")
    print("=" * 40)
    
    files_to_check = [
        "src/playwright_mcp/__init__.py",
        "src/playwright_mcp/server.py",
        "test_server.py",
    ]
    
    passed = 0
    total = len(files_to_check)
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if check_syntax(file_path):
                passed += 1
        else:
            print(f"✗ {file_path} not found")
    
    print(f"\nSyntax check passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All syntax checks passed!")
        print("\nServer structure summary:")
        print("- FastMCP server with browser lifespan management")
        print("- 17 tools for browser automation:")
        print("  * Navigation: navigate, reload, go_back, go_forward")
        print("  * DOM interaction: click, type_text, fill, select_option, hover, scroll")
        print("  * Element discovery: query_selector, query_selector_all")
        print("  * Snapshotting: get_html, get_accessibility_snapshot, screenshot, pdf")
        print("  * Script execution: evaluate")
        print("- Configurable browser settings (headless/headed, browser type, timeout)")
        print("- Support for stdio and HTTP transports")
        return 0
    else:
        print("✗ Some syntax errors found.")
        return 1

if __name__ == "__main__":
    sys.exit(main())