"""Playwright MCP Server - Main server implementation."""

import asyncio
import base64
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import Context, FastMCP
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class BrowserState:
    """Browser state container."""
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page


class NavigationResult(BaseModel):
    """Navigation operation result."""
    success: bool
    url: str
    error: Optional[str] = None


class ElementQueryResult(BaseModel):
    """Element query result."""
    found: bool
    count: int
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class ScreenshotResult(BaseModel):
    """Screenshot result."""
    success: bool
    data: Optional[str] = None  # base64 encoded
    format: str = "png"
    error: Optional[str] = None


class PDFResult(BaseModel):
    """PDF generation result."""
    success: bool
    data: Optional[str] = None  # base64 encoded
    error: Optional[str] = None


class ScriptResult(BaseModel):
    """JavaScript execution result."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


class Config:
    """Server configuration."""
    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        timeout: int = 30000,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
    ):
        self.headless = headless
        self.browser_type = browser_type
        self.timeout = timeout
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height


# Global configuration
config = Config()


@asynccontextmanager
async def browser_lifespan(server: FastMCP) -> AsyncIterator[BrowserState]:
    """Manage browser lifecycle."""
    logger.info("Starting browser...")
    
    playwright = await async_playwright().start()
    
    try:
        # Launch browser
        if config.browser_type == "chromium":
            browser = await playwright.chromium.launch(headless=config.headless)
        elif config.browser_type == "firefox":
            browser = await playwright.firefox.launch(headless=config.headless)
        elif config.browser_type == "webkit":
            browser = await playwright.webkit.launch(headless=config.headless)
        else:
            raise ValueError(f"Unsupported browser type: {config.browser_type}")
        
        # Create context
        context = await browser.new_context(
            viewport={"width": config.viewport_width, "height": config.viewport_height}
        )
        
        # Create page
        page = await context.new_page()
        
        # Set default timeout
        page.set_default_timeout(config.timeout)
        
        state = BrowserState(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page
        )
        
        logger.info("Browser started successfully")
        yield state
        
    finally:
        logger.info("Shutting down browser...")
        try:
            await browser.close()
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
        
        try:
            await playwright.stop()
        except Exception as e:
            logger.error(f"Error stopping playwright: {e}")


# Create FastMCP server with browser lifespan
mcp = FastMCP("Playwright MCP Server", lifespan=browser_lifespan)


def get_browser_state(ctx: Context) -> BrowserState:
    """Get browser state from context."""
    return ctx.request_context.lifespan_context


# Navigation Tools
@mcp.tool()
async def navigate(url: str, ctx: Context) -> NavigationResult:
    """Navigate to a URL."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.goto(url)
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def reload(ctx: Context) -> NavigationResult:
    """Reload the current page."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.reload()
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def go_back(ctx: Context) -> NavigationResult:
    """Go back in browser history."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.go_back()
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def go_forward(ctx: Context) -> NavigationResult:
    """Go forward in browser history."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.go_forward()
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


# DOM Interaction Tools
@mcp.tool()
async def click(selector: str, ctx: Context) -> Dict[str, Any]:
    """Click an element using any Playwright selector."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.click(selector)
        return {"success": True, "selector": selector}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def type_text(selector: str, text: str, ctx: Context) -> Dict[str, Any]:
    """Type text into an element."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.type(selector, text)
        return {"success": True, "selector": selector, "text": text}
    except Exception as e:
        return {"success": False, "selector": selector, "text": text, "error": str(e)}


@mcp.tool()
async def fill(selector: str, value: str, ctx: Context) -> Dict[str, Any]:
    """Fill an input field."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.fill(selector, value)
        return {"success": True, "selector": selector, "value": value}
    except Exception as e:
        return {"success": False, "selector": selector, "value": value, "error": str(e)}


@mcp.tool()
async def select_option(selector: str, value: str, ctx: Context) -> Dict[str, Any]:
    """Select an option from a dropdown."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.select_option(selector, value)
        return {"success": True, "selector": selector, "value": value}
    except Exception as e:
        return {"success": False, "selector": selector, "value": value, "error": str(e)}


@mcp.tool()
async def hover(selector: str, ctx: Context) -> Dict[str, Any]:
    """Hover over an element."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.hover(selector)
        return {"success": True, "selector": selector}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def scroll(selector: str, ctx: Context, x: int = 0, y: int = 0) -> Dict[str, Any]:
    """Scroll an element or the page."""
    try:
        browser_state = get_browser_state(ctx)
        if selector:
            element = await browser_state.page.query_selector(selector)
            if element:
                await element.scroll_into_view_if_needed()
                return {"success": True, "selector": selector, "x": x, "y": y}
            else:
                return {"success": False, "selector": selector, "error": "Element not found"}
        else:
            await browser_state.page.mouse.wheel(x, y)
            return {"success": True, "selector": selector, "x": x, "y": y}
    except Exception as e:
        return {"success": False, "selector": selector, "x": x, "y": y, "error": str(e)}


# Element Discovery Tools
@mcp.tool()
async def query_selector(selector: str, ctx: Context) -> ElementQueryResult:
    """Query for a single element."""
    try:
        browser_state = get_browser_state(ctx)
        element = await browser_state.page.query_selector(selector)
        if element:
            # Get element attributes
            tag_name = await element.evaluate("el => el.tagName")
            text_content = await element.evaluate("el => el.textContent")
            attributes = await element.evaluate("el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))")
            
            element_info = {
                "tag_name": tag_name,
                "text_content": text_content,
                "attributes": attributes
            }
            
            return ElementQueryResult(
                found=True,
                count=1,
                elements=[element_info]
            )
        else:
            return ElementQueryResult(found=False, count=0)
    except Exception as e:
        return ElementQueryResult(found=False, count=0, error=str(e))


@mcp.tool()
async def query_selector_all(selector: str, ctx: Context) -> ElementQueryResult:
    """Query for all matching elements."""
    try:
        browser_state = get_browser_state(ctx)
        elements = await browser_state.page.query_selector_all(selector)
        
        elements_info = []
        for element in elements:
            tag_name = await element.evaluate("el => el.tagName")
            text_content = await element.evaluate("el => el.textContent")
            attributes = await element.evaluate("el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))")
            
            elements_info.append({
                "tag_name": tag_name,
                "text_content": text_content,
                "attributes": attributes
            })
        
        return ElementQueryResult(
            found=len(elements) > 0,
            count=len(elements),
            elements=elements_info
        )
    except Exception as e:
        return ElementQueryResult(found=False, count=0, error=str(e))


# Snapshotting Tools
@mcp.tool()
async def get_html(ctx: Context) -> Dict[str, Any]:
    """Get the full HTML of the current page."""
    try:
        browser_state = get_browser_state(ctx)
        html = await browser_state.page.content()
        return {"success": True, "html": html}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_accessibility_snapshot(ctx: Context) -> Dict[str, Any]:
    """Get the accessibility tree snapshot."""
    try:
        browser_state = get_browser_state(ctx)
        snapshot = await browser_state.page.accessibility.snapshot()
        return {"success": True, "snapshot": snapshot}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def screenshot(
    ctx: Context,
    selector: Optional[str] = None,
    full_page: bool = False
) -> ScreenshotResult:
    """Take a screenshot of the page or specific element."""
    try:
        browser_state = get_browser_state(ctx)
        
        if selector:
            element = await browser_state.page.query_selector(selector)
            if element:
                screenshot_bytes = await element.screenshot()
            else:
                return ScreenshotResult(success=False, error="Element not found")
        else:
            screenshot_bytes = await browser_state.page.screenshot(full_page=full_page)
        
        # Encode as base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return ScreenshotResult(
            success=True,
            data=screenshot_base64,
            format="png"
        )
    except Exception as e:
        return ScreenshotResult(success=False, error=str(e))


@mcp.tool()
async def pdf(ctx: Context) -> PDFResult:
    """Generate a PDF of the current page."""
    try:
        browser_state = get_browser_state(ctx)
        pdf_bytes = await browser_state.page.pdf()
        
        # Encode as base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return PDFResult(success=True, data=pdf_base64)
    except Exception as e:
        return PDFResult(success=False, error=str(e))


# Script Evaluation Tool
@mcp.tool()
async def evaluate(script: str, ctx: Context) -> ScriptResult:
    """Execute JavaScript in the page context."""
    try:
        browser_state = get_browser_state(ctx)
        result = await browser_state.page.evaluate(script)
        return ScriptResult(success=True, result=result)
    except Exception as e:
        return ScriptResult(success=False, error=str(e))


# Element State & Validation Tools
@mcp.tool()
async def is_visible(selector: str, ctx: Context) -> Dict[str, Any]:
    """Check if an element is visible."""
    try:
        browser_state = get_browser_state(ctx)
        element = await browser_state.page.query_selector(selector)
        if element:
            visible = await element.is_visible()
            return {"success": True, "selector": selector, "visible": visible}
        else:
            return {"success": False, "selector": selector, "error": "Element not found"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def is_enabled(selector: str, ctx: Context) -> Dict[str, Any]:
    """Check if an element is enabled."""
    try:
        browser_state = get_browser_state(ctx)
        element = await browser_state.page.query_selector(selector)
        if element:
            enabled = await element.is_enabled()
            return {"success": True, "selector": selector, "enabled": enabled}
        else:
            return {"success": False, "selector": selector, "error": "Element not found"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def wait_for_element(selector: str, ctx: Context, timeout: int = 30000) -> Dict[str, Any]:
    """Wait for an element to appear."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_selector(selector, timeout=timeout)
        return {"success": True, "selector": selector, "timeout": timeout}
    except Exception as e:
        return {"success": False, "selector": selector, "timeout": timeout, "error": str(e)}


@mcp.tool()
async def wait_for_load_state(state: str, ctx: Context, timeout: int = 30000) -> Dict[str, Any]:
    """Wait for page load state (domcontentloaded, load, networkidle)."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_load_state(state, timeout=timeout)
        return {"success": True, "state": state, "timeout": timeout}
    except Exception as e:
        return {"success": False, "state": state, "timeout": timeout, "error": str(e)}


# Form & Input Handling Tools
@mcp.tool()
async def clear_text(selector: str, ctx: Context) -> Dict[str, Any]:
    """Clear text from an input field."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.fill(selector, "")
        return {"success": True, "selector": selector}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def check_checkbox(selector: str, ctx: Context) -> Dict[str, Any]:
    """Check a checkbox."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.check(selector)
        return {"success": True, "selector": selector, "action": "checked"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def uncheck_checkbox(selector: str, ctx: Context) -> Dict[str, Any]:
    """Uncheck a checkbox."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.uncheck(selector)
        return {"success": True, "selector": selector, "action": "unchecked"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def upload_file(selector: str, file_path: str, ctx: Context) -> Dict[str, Any]:
    """Upload a file to a file input."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.set_input_files(selector, file_path)
        return {"success": True, "selector": selector, "file_path": file_path}
    except Exception as e:
        return {"success": False, "selector": selector, "file_path": file_path, "error": str(e)}


@mcp.tool()
async def press_key(key: str, ctx: Context) -> Dict[str, Any]:
    """Press a keyboard key."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.keyboard.press(key)
        return {"success": True, "key": key}
    except Exception as e:
        return {"success": False, "key": key, "error": str(e)}


# Advanced Navigation Tools
@mcp.tool()
async def wait_for_url(url_pattern: str, ctx: Context, timeout: int = 30000) -> Dict[str, Any]:
    """Wait for URL to match a pattern."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_url(url_pattern, timeout=timeout)
        current_url = browser_state.page.url
        return {"success": True, "url_pattern": url_pattern, "current_url": current_url, "timeout": timeout}
    except Exception as e:
        return {"success": False, "url_pattern": url_pattern, "timeout": timeout, "error": str(e)}


@mcp.tool()
async def set_viewport_size(width: int, height: int, ctx: Context) -> Dict[str, Any]:
    """Set the viewport size."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.set_viewport_size({"width": width, "height": height})
        return {"success": True, "width": width, "height": height}
    except Exception as e:
        return {"success": False, "width": width, "height": height, "error": str(e)}


# Element Discovery & Analysis Tools
@mcp.tool()
async def get_element_bounding_box(selector: str, ctx: Context) -> Dict[str, Any]:
    """Get element position and size."""
    try:
        browser_state = get_browser_state(ctx)
        element = await browser_state.page.query_selector(selector)
        if element:
            box = await element.bounding_box()
            return {"success": True, "selector": selector, "bounding_box": box}
        else:
            return {"success": False, "selector": selector, "error": "Element not found"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def get_element_attributes(selector: str, ctx: Context) -> Dict[str, Any]:
    """Get all attributes of an element."""
    try:
        browser_state = get_browser_state(ctx)
        element = await browser_state.page.query_selector(selector)
        if element:
            attributes = await element.evaluate("el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))")
            return {"success": True, "selector": selector, "attributes": attributes}
        else:
            return {"success": False, "selector": selector, "error": "Element not found"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def get_computed_style(selector: str, property: str, ctx: Context) -> Dict[str, Any]:
    """Get computed CSS style property."""
    try:
        browser_state = get_browser_state(ctx)
        element = await browser_state.page.query_selector(selector)
        if element:
            style_value = await element.evaluate(f"el => getComputedStyle(el).{property}")
            return {"success": True, "selector": selector, "property": property, "value": style_value}
        else:
            return {"success": False, "selector": selector, "error": "Element not found"}
    except Exception as e:
        return {"success": False, "selector": selector, "property": property, "error": str(e)}


# Network & Debugging Tools
@mcp.tool()
async def wait_for_network_idle(ctx: Context, timeout: int = 30000) -> Dict[str, Any]:
    """Wait for network to be idle."""
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_load_state("networkidle", timeout=timeout)
        return {"success": True, "timeout": timeout}
    except Exception as e:
        return {"success": False, "timeout": timeout, "error": str(e)}


@mcp.tool()
async def get_page_errors(ctx: Context) -> Dict[str, Any]:
    """Get JavaScript errors from the page (requires setup during navigation)."""
    try:
        browser_state = get_browser_state(ctx)
        # Note: This would require setting up error listeners during browser initialization
        # For now, we'll evaluate to check for any stored errors
        errors = await browser_state.page.evaluate("""
            () => {
                if (window.pageErrors) {
                    return window.pageErrors;
                }
                return [];
            }
        """)
        return {"success": True, "errors": errors}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_console_logs(ctx: Context) -> Dict[str, Any]:
    """Get console logs from the page (requires setup during navigation)."""
    try:
        browser_state = get_browser_state(ctx)
        # Note: This would require setting up console listeners during browser initialization
        # For now, we'll return a message about setup requirements
        logs = await browser_state.page.evaluate("""
            () => {
                if (window.consoleLogs) {
                    return window.consoleLogs;
                }
                return [];
            }
        """)
        return {"success": True, "logs": logs, "note": "Console logging requires browser setup with listeners"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Playwright MCP Server")
    parser.add_argument("transport", choices=["stdio", "http"], help="Transport type")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    parser.add_argument("--headed", action="store_true", help="Run in headed mode")
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], 
                       default="chromium", help="Browser type")
    parser.add_argument("--timeout", type=int, default=30000, help="Default timeout (ms)")
    
    args = parser.parse_args()
    
    # Update global configuration
    config.headless = not args.headed
    config.browser_type = args.browser
    config.timeout = args.timeout
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the server using FastMCP's run method with transport
    if args.transport == "stdio":
        mcp.run()
    else:
        # HTTP transport using StreamableHTTP
        import uvicorn
        app = mcp.streamable_http_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()