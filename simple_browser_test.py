#!/usr/bin/env python3
"""
Simple test using just Playwright (no MCP) to demonstrate browser automation.
This is a standalone test that shows what the MCP server would do.
"""

import asyncio
import json
import base64
from pathlib import Path

async def simple_browser_test():
    """Test browser automation with Playwright directly."""
    print("🎭 Simple Browser Test (Pure Playwright)")
    print("=" * 50)
    
    try:
        from playwright.async_api import async_playwright
        print("✅ Playwright imported successfully")
        
        # Start Playwright
        playwright = await async_playwright().start()
        
        try:
            # Launch browser in headed mode (visible)
            print("🚀 Launching browser (headed mode)...")
            browser = await playwright.chromium.launch(headless=False)
            
            # Create context and page
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()
            
            print("✅ Browser launched successfully")
            print("🌐 Browser window should be visible now")
            print()
            
            # Navigate to a page
            print("🔄 Step 1: Navigate to example.com")
            await page.goto("https://example.com")
            print(f"📍 Current URL: {page.url}")
            
            # Wait for page to load
            await asyncio.sleep(2)
            
            # Get page title
            print("🔄 Step 2: Get page title")
            title = await page.title()
            print(f"📋 Page title: {title}")
            
            # Get HTML content
            print("🔄 Step 3: Get HTML content")
            html = await page.content()
            html_snippet = html[:200] + "..." if len(html) > 200 else html
            print(f"📄 HTML (first 200 chars): {html_snippet}")
            
            # Take screenshot
            print("🔄 Step 4: Take screenshot")
            screenshot_path = Path(__file__).parent / "simple_test_screenshot.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"📸 Screenshot saved to: {screenshot_path}")
            
            # Get accessibility snapshot
            print("🔄 Step 5: Get accessibility snapshot")
            accessibility_snapshot = await page.accessibility.snapshot()
            a11y_path = Path(__file__).parent / "simple_test_accessibility.json"
            with open(a11y_path, "w") as f:
                json.dump(accessibility_snapshot, f, indent=2)
            print(f"♿ Accessibility snapshot saved to: {a11y_path}")
            
            # Execute JavaScript
            print("🔄 Step 6: Execute JavaScript")
            js_result = await page.evaluate("document.querySelector('h1')?.textContent")
            print(f"🔧 JavaScript result (h1 text): {js_result}")
            
            # Find elements
            print("🔄 Step 7: Find elements")
            links = await page.query_selector_all("a")
            print(f"🔗 Found {len(links)} links on the page")
            
            if links:
                first_link = links[0]
                link_text = await first_link.text_content()
                link_href = await first_link.get_attribute("href")
                print(f"   First link: '{link_text}' -> {link_href}")
            
            # Keep browser open for a few seconds
            print("⏳ Keeping browser open for 5 seconds...")
            await asyncio.sleep(5)
            
            print("✅ All tests completed successfully!")
            
        finally:
            # Clean up
            print("🧹 Closing browser...")
            await browser.close()
            await playwright.stop()
            
    except ImportError:
        print("❌ Playwright not installed!")
        print("💡 Install it with:")
        print("   pip install playwright")
        print("   playwright install")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("🧪 Simple Browser Test - Pure Playwright")
    print("This will open a visible browser and navigate to example.com")
    print()
    
    input("Press Enter to start the test (or Ctrl+C to cancel)...")
    
    asyncio.run(simple_browser_test())

if __name__ == "__main__":
    main()