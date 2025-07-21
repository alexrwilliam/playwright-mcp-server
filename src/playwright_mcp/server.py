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
    """Navigate the browser to a specified URL.
    
    This tool loads a new page in the browser. It supports all standard URL formats
    including HTTP, HTTPS, file:// URLs, and data URLs.
    
    Args:
        url: The URL to navigate to (e.g., "https://example.com", "file:///path/to/file.html")
        ctx: MCP context containing the browser state
    
    Returns:
        NavigationResult with success status, final URL (after redirects), and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.goto(url)
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def reload(ctx: Context) -> NavigationResult:
    """Reload the current page in the browser.
    
    This tool refreshes the current page, equivalent to pressing F5 or clicking
    the browser's reload button. All page state will be reset.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        NavigationResult with success status, current URL, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.reload()
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def go_back(ctx: Context) -> NavigationResult:
    """Navigate back to the previous page in browser history.
    
    This tool moves back one step in the browser's navigation history,
    equivalent to clicking the browser's back button.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        NavigationResult with success status, current URL after navigation, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.go_back()
        current_url = browser_state.page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def go_forward(ctx: Context) -> NavigationResult:
    """Navigate forward to the next page in browser history.
    
    This tool moves forward one step in the browser's navigation history,
    equivalent to clicking the browser's forward button. Only works if there
    is a forward history (i.e., you've previously gone back).
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        NavigationResult with success status, current URL after navigation, and any errors
    """
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
    """Click an element on the page using a Playwright selector.
    
    This tool performs a left-click on the first element that matches the selector.
    Supports CSS selectors, text content, accessibility labels, and other Playwright selectors.
    
    Args:
        selector: Playwright selector to identify the element (e.g., "#button-id", "text=Click me", "[aria-label=Submit]")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector used, and any error messages
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.click(selector)
        return {"success": True, "selector": selector}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def type_text(selector: str, text: str, ctx: Context) -> Dict[str, Any]:
    """Type text into an input element character by character.
    
    This tool simulates human typing by sending individual keystrokes to the element.
    It does not clear existing text - use fill() for that behavior.
    
    Args:
        selector: Playwright selector for the input element (e.g., "input[name=username]", "#search-box")
        text: Text to type into the element
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, text typed, and any error messages
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.type(selector, text)
        return {"success": True, "selector": selector, "text": text}
    except Exception as e:
        return {"success": False, "selector": selector, "text": text, "error": str(e)}


@mcp.tool()
async def fill(selector: str, value: str, ctx: Context) -> Dict[str, Any]:
    """Fill an input field with text, replacing any existing content.
    
    This tool clears the existing text and sets the new value in one operation.
    More efficient than type_text for replacing content entirely.
    
    Args:
        selector: Playwright selector for the input element (e.g., "input[type=email]", "textarea")
        value: Text value to set in the input field
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, value set, and any error messages
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.fill(selector, value)
        return {"success": True, "selector": selector, "value": value}
    except Exception as e:
        return {"success": False, "selector": selector, "value": value, "error": str(e)}


@mcp.tool()
async def select_option(selector: str, value: str, ctx: Context) -> Dict[str, Any]:
    """Select an option from a dropdown/select element.
    
    This tool selects an option by value, label, or index from a <select> element.
    
    Args:
        selector: Playwright selector for the select element (e.g., "select[name=country]", "#dropdown")
        value: Option to select - can be the value attribute, visible text, or index (e.g., "US", "United States", "0")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, selected value, and any error messages
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.select_option(selector, value)
        return {"success": True, "selector": selector, "value": value}
    except Exception as e:
        return {"success": False, "selector": selector, "value": value, "error": str(e)}


@mcp.tool()
async def hover(selector: str, ctx: Context) -> Dict[str, Any]:
    """Move the mouse over an element to trigger hover effects.
    
    This tool simulates hovering the mouse cursor over an element, which can
    reveal tooltips, dropdown menus, or other hover-triggered content.
    
    Args:
        selector: Playwright selector for the element to hover over (e.g., ".menu-item", "[title=Help]")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, and any error messages
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.hover(selector)
        return {"success": True, "selector": selector}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def scroll(selector: str, ctx: Context, x: int = 0, y: int = 0) -> Dict[str, Any]:
    """Scroll an element into view or scroll the page by specified amounts.
    
    If a selector is provided, scrolls that element into view. If selector is empty,
    performs a page scroll by the specified x and y pixel amounts.
    
    Args:
        selector: Playwright selector for element to scroll into view, or empty string for page scroll
        ctx: MCP context containing the browser state
        x: Horizontal scroll amount in pixels (for page scroll)
        y: Vertical scroll amount in pixels (for page scroll)
    
    Returns:
        Dict with success status, selector, scroll amounts, and any error messages
    """
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
    """Find and return information about the first element matching a selector.
    
    This tool locates a single element on the page and returns detailed information
    including its tag name, text content, and all attributes.
    
    Args:
        selector: Playwright selector to find the element (e.g., "#main-header", "button:has-text('Submit')", "[data-testid=login]")
        ctx: MCP context containing the browser state
    
    Returns:
        ElementQueryResult with found status, element details (tag, text, attributes), and any errors
    """
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
    """Find and return information about all elements matching a selector.
    
    This tool locates all elements on the page that match the selector and returns
    detailed information for each one including tag names, text content, and attributes.
    
    Args:
        selector: Playwright selector to find elements (e.g., ".nav-item", "input[type=checkbox]", "li")
        ctx: MCP context containing the browser state
    
    Returns:
        ElementQueryResult with found status, count, array of element details, and any errors
    """
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
    """Retrieve the complete HTML source of the current page.
    
    This tool returns the full HTML content including the DOCTYPE, head, and body
    sections. Useful for analyzing page structure or saving page content.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status and the complete HTML source as a string
    """
    try:
        browser_state = get_browser_state(ctx)
        html = await browser_state.page.content()
        return {"success": True, "html": html}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_accessibility_snapshot(ctx: Context) -> Dict[str, Any]:
    """Capture the accessibility tree structure of the current page.
    
    This tool returns the accessibility tree used by screen readers and other
    assistive technologies, including roles, names, and hierarchical structure.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status and the accessibility tree snapshot structure
    """
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
    """Capture a screenshot of the page or a specific element.
    
    This tool generates a PNG image of either the current viewport, the entire page,
    or a specific element. The image is returned as base64-encoded data.
    
    Args:
        ctx: MCP context containing the browser state
        selector: Optional Playwright selector for a specific element to screenshot
        full_page: If True, captures the entire page including content below the fold
    
    Returns:
        ScreenshotResult with success status, base64-encoded PNG data, format, and any errors
    """
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
    """Generate a PDF document from the current page.
    
    This tool converts the current page to a PDF format, useful for saving
    or printing web content. Only works with Chromium-based browsers.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        PDFResult with success status, base64-encoded PDF data, and any errors
    """
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
    """Execute JavaScript code in the browser page context.
    
    This tool runs arbitrary JavaScript code within the page's context,
    allowing access to DOM, window objects, and page variables.
    
    Args:
        script: JavaScript code to execute (e.g., "document.title", "window.scrollTo(0, 100)")
        ctx: MCP context containing the browser state
    
    Returns:
        ScriptResult with success status, execution result (if any), and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        result = await browser_state.page.evaluate(script)
        return ScriptResult(success=True, result=result)
    except Exception as e:
        return ScriptResult(success=False, error=str(e))


# Element State & Validation Tools
@mcp.tool()
async def is_visible(selector: str, ctx: Context) -> Dict[str, Any]:
    """Check whether an element is visible on the page.
    
    This tool determines if an element is currently visible to users, taking into
    account CSS visibility, display properties, and whether it's within the viewport.
    
    Args:
        selector: Playwright selector for the element to check (e.g., "#modal", ".hidden-element")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, visibility boolean, and any errors
    """
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
    """Check whether an element is enabled and can be interacted with.
    
    This tool determines if form elements like buttons, inputs, and selects
    are enabled (not disabled) and can receive user interactions.
    
    Args:
        selector: Playwright selector for the element to check (e.g., "button[type=submit]", "input[name=email]")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, enabled boolean, and any errors
    """
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
    """Wait for an element to appear in the DOM.
    
    This tool waits until an element matching the selector becomes available,
    useful for handling dynamic content that loads after page initialization.
    
    Args:
        selector: Playwright selector for the element to wait for (e.g., ".loading-complete", "[data-loaded=true]")
        ctx: MCP context containing the browser state
        timeout: Maximum wait time in milliseconds (default: 30000)
    
    Returns:
        Dict with success status, selector, timeout value, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_selector(selector, timeout=timeout)
        return {"success": True, "selector": selector, "timeout": timeout}
    except Exception as e:
        return {"success": False, "selector": selector, "timeout": timeout, "error": str(e)}


@mcp.tool()
async def wait_for_load_state(state: str, ctx: Context, timeout: int = 30000) -> Dict[str, Any]:
    """Wait for the page to reach a specific loading state.
    
    This tool waits for different stages of page loading to complete, ensuring
    content is ready before proceeding with interactions.
    
    Args:
        state: Loading state to wait for - "domcontentloaded" (DOM ready), "load" (all resources), or "networkidle" (no requests for 500ms)
        ctx: MCP context containing the browser state
        timeout: Maximum wait time in milliseconds (default: 30000)
    
    Returns:
        Dict with success status, target state, timeout value, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_load_state(state, timeout=timeout)
        return {"success": True, "state": state, "timeout": timeout}
    except Exception as e:
        return {"success": False, "state": state, "timeout": timeout, "error": str(e)}


# Form & Input Handling Tools
@mcp.tool()
async def clear_text(selector: str, ctx: Context) -> Dict[str, Any]:
    """Clear all text content from an input field.
    
    This tool removes all existing text from text inputs, textareas, and other
    editable fields, leaving them empty.
    
    Args:
        selector: Playwright selector for the input element (e.g., "input[name=message]", "#comment-box")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.fill(selector, "")
        return {"success": True, "selector": selector}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def check_checkbox(selector: str, ctx: Context) -> Dict[str, Any]:
    """Check a checkbox or radio button element.
    
    This tool selects a checkbox or radio button, setting it to the checked state.
    If already checked, the operation has no effect.
    
    Args:
        selector: Playwright selector for the checkbox/radio element (e.g., "input[type=checkbox][name=agree]", "#newsletter")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, action performed, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.check(selector)
        return {"success": True, "selector": selector, "action": "checked"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def uncheck_checkbox(selector: str, ctx: Context) -> Dict[str, Any]:
    """Uncheck a checkbox element.
    
    This tool deselects a checkbox, setting it to the unchecked state.
    If already unchecked, the operation has no effect.
    
    Args:
        selector: Playwright selector for the checkbox element (e.g., "input[type=checkbox][name=notifications]", ".privacy-checkbox")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, action performed, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.uncheck(selector)
        return {"success": True, "selector": selector, "action": "unchecked"}
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def upload_file(selector: str, file_path: str, ctx: Context) -> Dict[str, Any]:
    """Upload a file through a file input element.
    
    This tool selects and uploads a file from the local filesystem to a
    file input element on the page.
    
    Args:
        selector: Playwright selector for the file input element (e.g., "input[type=file]", "#avatar-upload")
        file_path: Absolute path to the file to upload (e.g., "/path/to/document.pdf")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, file path, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.set_input_files(selector, file_path)
        return {"success": True, "selector": selector, "file_path": file_path}
    except Exception as e:
        return {"success": False, "selector": selector, "file_path": file_path, "error": str(e)}


@mcp.tool()
async def press_key(key: str, ctx: Context) -> Dict[str, Any]:
    """Simulate pressing a keyboard key.
    
    This tool sends a key press event to the page, useful for shortcuts,
    navigation keys, or special key combinations.
    
    Args:
        key: Key to press (e.g., "Enter", "Escape", "Tab", "Control+s", "ArrowDown")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, key pressed, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.keyboard.press(key)
        return {"success": True, "key": key}
    except Exception as e:
        return {"success": False, "key": key, "error": str(e)}


# Advanced Navigation Tools
@mcp.tool()
async def wait_for_url(url_pattern: str, ctx: Context, timeout: int = 30000) -> Dict[str, Any]:
    """Wait for the browser URL to match a specific pattern.
    
    This tool waits until the current page URL matches the provided pattern,
    useful for handling redirects or navigation completion.
    
    Args:
        url_pattern: URL pattern to match - can be exact URL, glob pattern, or regex (e.g., "**/dashboard", "https://example.com/success")
        ctx: MCP context containing the browser state
        timeout: Maximum wait time in milliseconds (default: 30000)
    
    Returns:
        Dict with success status, URL pattern, current URL, timeout, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_url(url_pattern, timeout=timeout)
        current_url = browser_state.page.url
        return {"success": True, "url_pattern": url_pattern, "current_url": current_url, "timeout": timeout}
    except Exception as e:
        return {"success": False, "url_pattern": url_pattern, "timeout": timeout, "error": str(e)}


@mcp.tool()
async def set_viewport_size(width: int, height: int, ctx: Context) -> Dict[str, Any]:
    """Change the browser viewport dimensions.
    
    This tool resizes the browser's viewport to simulate different screen sizes,
    useful for testing responsive designs or mobile layouts.
    
    Args:
        width: Viewport width in pixels (e.g., 1920, 768, 375)
        height: Viewport height in pixels (e.g., 1080, 1024, 667)
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, new dimensions, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.set_viewport_size({"width": width, "height": height})
        return {"success": True, "width": width, "height": height}
    except Exception as e:
        return {"success": False, "width": width, "height": height, "error": str(e)}


# Element Discovery & Analysis Tools
@mcp.tool()
async def get_element_bounding_box(selector: str, ctx: Context) -> Dict[str, Any]:
    """Get the position and dimensions of an element.
    
    This tool returns the bounding box coordinates and size of an element,
    useful for layout analysis or positioning calculations.
    
    Args:
        selector: Playwright selector for the element (e.g., "#main-content", ".sidebar")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, bounding box (x, y, width, height), and any errors
    """
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
    """Retrieve all HTML attributes of an element.
    
    This tool returns a dictionary of all attributes and their values for
    the specified element, useful for inspecting element properties.
    
    Args:
        selector: Playwright selector for the element (e.g., "img[alt]", "a.external-link")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, attributes dictionary, and any errors
    """
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
    """Get the computed CSS value for a specific style property of an element.
    
    This tool retrieves the final computed CSS value after all stylesheets
    and inheritance rules have been applied.
    
    Args:
        selector: Playwright selector for the element (e.g., ".header", "#main-nav")
        property: CSS property name (e.g., "color", "fontSize", "display", "margin-top")
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, selector, property name, computed value, and any errors
    """
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
    """Wait for network activity to become idle.
    
    This tool waits until there are no network requests for at least 500ms,
    indicating that dynamic content loading has completed.
    
    Args:
        ctx: MCP context containing the browser state
        timeout: Maximum wait time in milliseconds (default: 30000)
    
    Returns:
        Dict with success status, timeout value, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.page.wait_for_load_state("networkidle", timeout=timeout)
        return {"success": True, "timeout": timeout}
    except Exception as e:
        return {"success": False, "timeout": timeout, "error": str(e)}


@mcp.tool()
async def get_page_errors(ctx: Context) -> Dict[str, Any]:
    """Retrieve JavaScript errors that occurred on the page.
    
    This tool returns any JavaScript errors that have been captured during
    page execution. Note: Error collection requires browser setup with listeners.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, array of error messages, and any errors
    """
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
    """Retrieve console log messages from the page.
    
    This tool returns console.log, console.error, and other console messages
    that occurred during page execution. Note: Log collection requires browser setup.
    
    Args:
        ctx: MCP context containing the browser state
    
    Returns:
        Dict with success status, array of console messages, setup note, and any errors
    """
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