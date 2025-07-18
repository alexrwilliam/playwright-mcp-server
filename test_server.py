#!/usr/bin/env python3
"""Test script to verify the Playwright MCP server structure."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from playwright_mcp.server import mcp
        print("✓ Successfully imported server module")
        
        # Test that the server has tools
        print(f"✓ Server has {len(mcp._tools)} tools registered")
        
        # List all tools
        print("\nRegistered tools:")
        for tool_name in mcp._tools:
            print(f"  - {tool_name}")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_server_structure():
    """Test server structure and configuration."""
    try:
        from playwright_mcp.server import config, Config
        print(f"✓ Server configuration loaded")
        print(f"  - Headless: {config.headless}")
        print(f"  - Browser: {config.browser_type}")
        print(f"  - Timeout: {config.timeout}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Playwright MCP Server...")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_server_structure,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed! Server is ready.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())