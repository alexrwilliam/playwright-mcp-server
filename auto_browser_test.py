#!/usr/bin/env python3
"""
Automatic browser test - runs without user input.
This will open a visible browser, navigate to example.com, and take snapshots.
"""

import asyncio
import json
from pathlib import Path

async def auto_browser_test():
    """Test browser automation with Playwright directly."""
    print("🎭 Automatic Browser Test - Opening Real Browser!")
    print("=" * 60)
    
    try:
        from playwright.async_api import async_playwright
        print("✅ Playwright imported successfully")
        
        # Start Playwright
        playwright = await async_playwright().start()
        
        try:
            # Launch browser in headed mode (visible)
            print("🚀 Launching browser in headed mode...")
            browser = await playwright.chromium.launch(headless=False)
            
            # Create context and page
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()
            
            print("✅ Browser launched successfully")
            print("🌐 Browser window is now visible!")
            print()
            
            # Navigate to a page
            print("🔄 Step 1: Navigate to example.com")
            await page.goto("https://example.com")
            print(f"📍 Current URL: {page.url}")
            
            # Wait for page to load
            print("⏳ Waiting 2 seconds for page to fully load...")
            await asyncio.sleep(2)
            
            # Get page title
            print("🔄 Step 2: Get page title")
            title = await page.title()
            print(f"📋 Page title: '{title}'")
            
            # Get HTML content
            print("🔄 Step 3: Get HTML content")
            html = await page.content()
            html_snippet = html[:200] + "..." if len(html) > 200 else html
            print(f"📄 HTML (first 200 chars): {html_snippet}")
            
            # Take screenshot
            print("🔄 Step 4: Take full-page screenshot")
            screenshot_path = Path(__file__).parent / "auto_test_screenshot.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"📸 Screenshot saved to: {screenshot_path}")
            
            # Get accessibility snapshot
            print("🔄 Step 5: Get accessibility snapshot")
            accessibility_snapshot = await page.accessibility.snapshot()
            a11y_path = Path(__file__).parent / "auto_test_accessibility.json"
            with open(a11y_path, "w") as f:
                json.dump(accessibility_snapshot, f, indent=2)
            print(f"♿ Accessibility snapshot saved to: {a11y_path}")
            
            # Execute JavaScript
            print("🔄 Step 6: Execute JavaScript")
            js_result = await page.evaluate("document.title")
            print(f"🔧 JavaScript result (document.title): '{js_result}'")
            
            # Get page dimensions
            print("🔄 Step 7: Get page dimensions")
            dimensions = await page.evaluate("""
                () => ({
                    width: document.body.scrollWidth,
                    height: document.body.scrollHeight,
                    viewportWidth: window.innerWidth,
                    viewportHeight: window.innerHeight
                })
            """)
            print(f"📐 Page dimensions: {dimensions}")
            
            # Find elements
            print("🔄 Step 8: Find elements on page")
            
            # Find heading
            heading = await page.query_selector("h1")
            if heading:
                heading_text = await heading.text_content()
                print(f"🔍 Found H1: '{heading_text}'")
            
            # Find all links
            links = await page.query_selector_all("a")
            print(f"🔗 Found {len(links)} links on the page")
            
            if links:
                for i, link in enumerate(links):
                    link_text = await link.text_content()
                    link_href = await link.get_attribute("href")
                    print(f"   Link {i+1}: '{link_text}' -> {link_href}")
            
            # Find all paragraphs
            paragraphs = await page.query_selector_all("p")
            print(f"📝 Found {len(paragraphs)} paragraphs")
            
            if paragraphs:
                for i, p in enumerate(paragraphs):
                    p_text = await p.text_content()
                    snippet = p_text[:100] + "..." if len(p_text) > 100 else p_text
                    print(f"   Paragraph {i+1}: {snippet}")
            
            # Show what files were created
            print("\n📁 Files created:")
            if screenshot_path.exists():
                size = screenshot_path.stat().st_size
                print(f"   📸 {screenshot_path} ({size:,} bytes)")
            if a11y_path.exists():
                size = a11y_path.stat().st_size
                print(f"   ♿ {a11y_path} ({size:,} bytes)")
            
            # Keep browser open for a moment
            print("\n⏳ Keeping browser open for 5 seconds so you can see it...")
            await asyncio.sleep(5)
            
            print("\n✅ All tests completed successfully!")
            print("🎉 This demonstrates exactly what the MCP server would do!")
            
        finally:
            # Clean up
            print("🧹 Closing browser...")
            await browser.close()
            await playwright.stop()
            print("✅ Cleanup completed")
            
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
    print("🧪 Automatic Browser Test Starting...")
    print("This will open a visible browser window and navigate to example.com")
    print()
    
    asyncio.run(auto_browser_test())
    
    print("\n🎯 What just happened:")
    print("   • Real browser opened in headed mode (visible)")
    print("   • Navigated to example.com")
    print("   • Extracted page title, HTML, and elements")
    print("   • Took full-page screenshot")
    print("   • Generated accessibility snapshot")
    print("   • Executed JavaScript")
    print("   • All data is raw Playwright output")
    print("\n🚀 This is exactly what your MCP server does!")

if __name__ == "__main__":
    main()