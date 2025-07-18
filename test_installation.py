#!/usr/bin/env python3
"""
Test installation and guide for setting up the Playwright MCP server.
This script checks dependencies and provides installation guidance.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {module_name} - Not installed")
        return False

def check_playwright_browsers():
    """Check if Playwright browsers are installed."""
    try:
        import playwright
        # Try to check if browsers are installed
        # This is a simple check - in reality we'd need to run playwright install --dry-run
        print("✅ Playwright module found")
        return True
    except ImportError:
        print("❌ Playwright module not found")
        return False

def run_installation_guide():
    """Provide step-by-step installation guide."""
    print("\n🔧 Installation Guide")
    print("=" * 30)
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  You're not in a virtual environment. Recommended steps:")
        print("   1. Create virtual environment:")
        print("      python3 -m venv venv")
        print("   2. Activate it:")
        print("      source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   3. Then install dependencies:")
    else:
        print("✅ You're in a virtual environment")
        print("   Install dependencies:")
    
    print("      pip install playwright")
    print("      playwright install")
    print("      pip install mcp pydantic asyncio-throttle")
    print()
    
    print("📋 After installation, you can:")
    print("   1. Run simple test:")
    print("      python3 simple_browser_test.py")
    print("   2. Run MCP server test:")
    print("      python3 test_real_browser.py")
    print("   3. Start the MCP server:")
    print("      python3 src/playwright_mcp/server.py stdio --headed")

def attempt_simple_install():
    """Attempt to install just playwright for testing."""
    print("\n🚀 Attempting to install Playwright for testing...")
    
    try:
        # Try to install playwright
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
        print("✅ Playwright installed successfully")
        
        # Try to install browsers
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("✅ Chromium browser installed successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def main():
    """Main function to check installation and guide setup."""
    print("🎭 Playwright MCP Server - Installation Check")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    print("\n📦 Checking Dependencies:")
    print("-" * 25)
    
    # Check required modules
    modules_to_check = [
        "playwright",
        "mcp", 
        "pydantic",
        "asyncio"
    ]
    
    missing_modules = []
    for module in modules_to_check:
        if not check_module(module):
            missing_modules.append(module)
    
    # Check project structure
    print("\n📁 Checking Project Structure:")
    print("-" * 30)
    
    required_files = [
        "src/playwright_mcp/server.py",
        "src/playwright_mcp/__init__.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - Found")
        else:
            print(f"❌ {file_path} - Missing")
    
    # Provide recommendations
    print("\n🎯 Next Steps:")
    print("-" * 15)
    
    if missing_modules:
        print(f"❌ Missing modules: {', '.join(missing_modules)}")
        run_installation_guide()
        
        # Ask if user wants to try automatic installation
        if "playwright" in missing_modules:
            try:
                response = input("\n🤔 Would you like to try installing Playwright automatically? (y/n): ")
                if response.lower() in ['y', 'yes']:
                    if attempt_simple_install():
                        print("\n✅ Installation successful! You can now run:")
                        print("   python3 simple_browser_test.py")
                    else:
                        print("\n❌ Automatic installation failed. Please install manually.")
            except KeyboardInterrupt:
                print("\n⚠️  Installation cancelled.")
    else:
        print("✅ All dependencies are installed!")
        print("🎉 You can now run the tests:")
        print("   python3 simple_browser_test.py      # Simple Playwright test")
        print("   python3 test_real_browser.py        # Full MCP server test")
        print("   python3 src/playwright_mcp/server.py stdio --headed  # Start MCP server")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())