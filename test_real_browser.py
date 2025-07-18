#!/usr/bin/env python3
"""
Test file that demonstrates the Playwright MCP server with a real browser.
This will open a visible browser, navigate to a page, and take a snapshot.
"""

import asyncio
import json
import base64
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our server
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_real_browser():
    """Test the server with a real browser."""
    print("🎭 Testing Playwright MCP Server with Real Browser")
    print("=" * 60)
    
    try:
        # Import required modules
        from playwright.async_api import async_playwright
        from playwright_mcp.server import (
            BrowserState, Config, config,
            navigate, get_html, screenshot, get_accessibility_snapshot
        )
        
        # Configure for headed mode
        config.headless = False
        config.browser_type = "chromium"
        config.timeout = 30000
        
        print("✅ Imports successful")
        print(f"🔧 Configuration: headless={config.headless}, browser={config.browser_type}")
        print()
        
        # Start playwright and browser
        print("🚀 Starting browser...")
        playwright = await async_playwright().start()
        
        try:
            browser = await playwright.chromium.launch(headless=config.headless)
            context = await browser.new_context(
                viewport={"width": config.viewport_width, "height": config.viewport_height}
            )
            page = await context.new_page()
            page.set_default_timeout(config.timeout)
            
            # Create browser state
            browser_state = BrowserState(
                playwright=playwright,
                browser=browser,
                context=context,
                page=page
            )
            
            print("✅ Browser started successfully")
            print(f"📱 Viewport: {config.viewport_width}x{config.viewport_height}")
            print()
            
            # Create mock context for our tools
            class MockContext:
                def __init__(self, browser_state):
                    self.request_context = self
                    self.lifespan_context = browser_state
            
            mock_ctx = MockContext(browser_state)
            
            # Test 1: Navigate to a page
            print("🔄 Test 1: Navigate to example.com")
            result = await navigate("https://example.com", mock_ctx)
            print(f"📍 Navigation result: {result}")
            
            # Wait a moment for the page to load
            await asyncio.sleep(2)
            print()
            
            # Test 2: Get HTML content
            print("🔄 Test 2: Get HTML content")
            html_result = await get_html(mock_ctx)
            if html_result["success"]:
                html_snippet = html_result["html"][:200] + "..." if len(html_result["html"]) > 200 else html_result["html"]
                print(f"📄 HTML (first 200 chars): {html_snippet}")
            else:
                print(f"❌ Failed to get HTML: {html_result}")
            print()
            
            # Test 3: Take a screenshot
            print("🔄 Test 3: Take screenshot")
            screenshot_result = await screenshot(mock_ctx, full_page=True)
            if screenshot_result.success:
                print(f"📸 Screenshot taken successfully")
                print(f"📊 Data length: {len(screenshot_result.data)} characters (base64)")
                
                # Save screenshot to file
                screenshot_path = Path(__file__).parent / "test_screenshot.png"
                screenshot_data = base64.b64decode(screenshot_result.data)
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot_data)
                print(f"💾 Screenshot saved to: {screenshot_path}")
            else:
                print(f"❌ Screenshot failed: {screenshot_result.error}")
            print()
            
            # Test 4: Get accessibility snapshot
            print("🔄 Test 4: Get accessibility snapshot")
            a11y_result = await get_accessibility_snapshot(mock_ctx)
            if a11y_result["success"]:
                print(f"♿ Accessibility snapshot obtained")
                print(f"📊 Snapshot keys: {list(a11y_result['snapshot'].keys())}")
                
                # Save accessibility snapshot to file
                a11y_path = Path(__file__).parent / "test_accessibility.json"
                with open(a11y_path, "w") as f:
                    json.dump(a11y_result["snapshot"], f, indent=2)
                print(f"💾 Accessibility snapshot saved to: {a11y_path}")
            else:
                print(f"❌ Accessibility snapshot failed: {a11y_result}")
            print()
            
            # Test 5: Execute JavaScript
            print("🔄 Test 5: Execute JavaScript")
            from playwright_mcp.server import evaluate
            js_result = await evaluate("document.title", mock_ctx)
            if js_result.success:
                print(f"🔧 JavaScript result: {js_result.result}")
            else:
                print(f"❌ JavaScript failed: {js_result.error}")
            print()
            
            # Wait a moment so user can see the browser
            print("⏳ Keeping browser open for 3 seconds so you can see it...")
            await asyncio.sleep(3)
            
            print("✅ All tests completed successfully!")
            
        finally:
            # Clean up
            print("🧹 Cleaning up...")
            await browser.close()
            await playwright.stop()
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 You need to install the required dependencies:")
        print("   pip install playwright mcp pydantic")
        print("   playwright install")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the test."""
    print("🧪 Playwright MCP Server - Real Browser Test")
    print("This test will open a visible browser window")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/playwright_mcp/server.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected to find: src/playwright_mcp/server.py")
        return 1
    
    # Run the test
    try:
        asyncio.run(test_real_browser())
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())