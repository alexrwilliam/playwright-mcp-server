#!/usr/bin/env python3
"""
Demo client to test the Playwright MCP server.
This simulates what an MCP client would do.
"""

import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the required dependencies for demo purposes
class MockFastMCP:
    def __init__(self, name, lifespan=None):
        self.name = name
        self.lifespan = lifespan
        self._tools = {}
        self._running = False
    
    def tool(self):
        def decorator(func):
            self._tools[func.__name__] = func
            return func
        return decorator
    
    def run(self, transport="stdio"):
        print(f"🚀 {self.name} would start with {transport} transport")
        print(f"📋 Registered {len(self._tools)} tools:")
        for name in self._tools:
            print(f"   • {name}")

# Mock MCP types
class MockContext:
    def __init__(self):
        self.request_context = self
        self.lifespan_context = MockBrowserState()

class MockBrowserState:
    def __init__(self):
        self.page = MockPage()

class MockPage:
    def __init__(self):
        self.url = "about:blank"
    
    async def goto(self, url):
        self.url = url
        print(f"🌐 Navigated to: {url}")
        return url
    
    async def reload(self):
        print(f"🔄 Reloaded page: {self.url}")
    
    async def content(self):
        return f"<html><head><title>Demo Page</title></head><body><h1>Hello from {self.url}</h1></body></html>"
    
    async def screenshot(self, full_page=False):
        print(f"📸 Taking {'full page' if full_page else 'viewport'} screenshot")
        return b"fake_screenshot_data"
    
    async def evaluate(self, script):
        print(f"🔧 Executing JavaScript: {script}")
        if script == "document.title":
            return "Demo Page"
        return "Script result"

# Patch the imports
import sys
class MockMCP:
    class server:
        class fastmcp:
            FastMCP = MockFastMCP
            Context = MockContext

class MockPlaywright:
    class async_api:
        pass

sys.modules['mcp'] = MockMCP()
sys.modules['mcp.server'] = MockMCP.server()
sys.modules['mcp.server.fastmcp'] = MockMCP.server.fastmcp()
sys.modules['playwright'] = MockPlaywright()
sys.modules['playwright.async_api'] = MockPlaywright.async_api()

async def demo_server():
    """Demonstrate the server in action."""
    print("=== Playwright MCP Server Demo ===\n")
    
    # Import and patch the server
    try:
        from playwright_mcp.server import mcp
        print(f"✅ Successfully loaded server: {mcp.name}")
        print(f"📦 Available tools: {list(mcp._tools.keys())}")
        print()
        
        # Create a mock context
        ctx = MockContext()
        
        # Demo 1: Navigate to a page
        print("🔍 Demo 1: Navigate to a website")
        navigate_func = mcp._tools['navigate']
        result = await navigate_func("https://example.com", ctx)
        print(f"Result: {result}")
        print()
        
        # Demo 2: Get page HTML
        print("🔍 Demo 2: Get page HTML")
        html_func = mcp._tools['get_html']
        result = await html_func(ctx)
        print(f"HTML snippet: {result['html'][:100]}...")
        print()
        
        # Demo 3: Take screenshot
        print("🔍 Demo 3: Take screenshot")
        screenshot_func = mcp._tools['screenshot']
        result = await screenshot_func(ctx, full_page=True)
        print(f"Screenshot result: {result}")
        print()
        
        # Demo 4: Execute JavaScript
        print("🔍 Demo 4: Execute JavaScript")
        eval_func = mcp._tools['evaluate']
        result = await eval_func("document.title", ctx)
        print(f"JavaScript result: {result}")
        print()
        
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def show_usage():
    """Show how to actually run the server."""
    print("\n=== How to Run the Real Server ===")
    print()
    print("1. Install dependencies (in a virtual environment):")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate")
    print("   pip install mcp playwright pydantic")
    print("   playwright install")
    print()
    print("2. Run the server:")
    print("   python3 src/playwright_mcp/server.py stdio")
    print("   # or")
    print("   python3 src/playwright_mcp/server.py http --port 8000")
    print()
    print("3. Test with MCP Inspector (requires uv):")
    print("   uv run mcp dev src/playwright_mcp/server.py")
    print()
    print("4. Connect from Claude Desktop:")
    print("   Add the server config to claude_desktop_config.json")
    print("   (see examples/claude_desktop_config.json)")

if __name__ == "__main__":
    print("🎭 Playwright MCP Server Demo Client")
    print("=" * 50)
    
    # Run the demo
    asyncio.run(demo_server())
    
    # Show actual usage instructions
    show_usage()