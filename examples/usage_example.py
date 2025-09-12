#!/usr/bin/env python3
"""Example usage of the Playwright MCP Server."""

import asyncio
import json
from typing import Dict, Any

# This example shows how you would use the MCP server tools
# In practice, these would be called through the MCP protocol

class MockContext:
    """Mock context for demonstration purposes."""
    def __init__(self):
        self.browser_state = None

async def demo_browser_automation():
    """Demonstrate browser automation capabilities."""
    print("=== Playwright MCP Server Demo ===\n")
    
    # Example tool calls that would be made through MCP
    examples = [
        {
            "tool": "navigate",
            "description": "Navigate to a website",
            "params": {"url": "https://example.com"},
            "expected_result": {
                "success": True,
                "url": "https://example.com"
            }
        },
        {
            "tool": "get_current_url",
            "description": "Get current page URL with parsed components",
            "params": {},
            "expected_result": {
                "success": True,
                "url": "https://example.com",
                "parsed_url": {
                    "scheme": "https",
                    "netloc": "example.com",
                    "hostname": "example.com",
                    "port": None,
                    "path": "/",
                    "fragment": "",
                    "query": ""
                },
                "query_params": {}
            }
        },
        {
            "tool": "query_selector",
            "description": "Find an element on the page",
            "params": {"selector": "h1"},
            "expected_result": {
                "found": True,
                "count": 1,
                "elements": [
                    {
                        "tag_name": "H1",
                        "text_content": "Example Domain",
                        "attributes": {}
                    }
                ]
            }
        },
        {
            "tool": "click",
            "description": "Click on an element",
            "params": {"selector": "a[href='https://www.iana.org/domains/example']"},
            "expected_result": {
                "success": True,
                "selector": "a[href='https://www.iana.org/domains/example']"
            }
        },
        {
            "tool": "screenshot",
            "description": "Take a screenshot of the page",
            "params": {"full_page": True},
            "expected_result": {
                "success": True,
                "data": "iVBORw0KGgoAAAANSUhEUgAA...",  # base64 encoded image
                "format": "png"
            }
        },
        {
            "tool": "evaluate",
            "description": "Execute JavaScript in the page",
            "params": {"script": "document.title"},
            "expected_result": {
                "success": True,
                "result": "Example Domain"
            }
        },
        {
            "tool": "get_html",
            "description": "Get the full HTML of the page",
            "params": {},
            "expected_result": {
                "success": True,
                "html": "<!doctype html><html>..."
            }
        },
        {
            "tool": "fill",
            "description": "Fill an input field",
            "params": {"selector": "input[type='text']", "value": "Hello World"},
            "expected_result": {
                "success": True,
                "selector": "input[type='text']",
                "value": "Hello World"
            }
        },
        {
            "tool": "pdf",
            "description": "Generate PDF of the page",
            "params": {},
            "expected_result": {
                "success": True,
                "data": "JVBERi0xLjQKJcfs..."  # base64 encoded PDF
            }
        },
        {
            "tool": "is_visible",
            "description": "Check if an element is visible",
            "params": {"selector": "h1"},
            "expected_result": {
                "success": True,
                "selector": "h1",
                "visible": True
            }
        },
        {
            "tool": "wait_for_element",
            "description": "Wait for an element to appear",
            "params": {"selector": ".dynamic-content", "timeout": 5000},
            "expected_result": {
                "success": True,
                "selector": ".dynamic-content",
                "timeout": 5000
            }
        },
        {
            "tool": "check_checkbox",
            "description": "Check a checkbox",
            "params": {"selector": "input[type='checkbox']"},
            "expected_result": {
                "success": True,
                "selector": "input[type='checkbox']",
                "action": "checked"
            }
        },
        {
            "tool": "press_key",
            "description": "Press a keyboard key",
            "params": {"key": "Enter"},
            "expected_result": {
                "success": True,
                "key": "Enter"
            }
        },
        {
            "tool": "get_element_bounding_box",
            "description": "Get element position and dimensions",
            "params": {"selector": "h1"},
            "expected_result": {
                "success": True,
                "selector": "h1",
                "bounding_box": {"x": 50, "y": 100, "width": 200, "height": 40}
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   Tool: {example['tool']}")
        print(f"   Parameters: {json.dumps(example['params'], indent=6)}")
        print(f"   Expected Result: {json.dumps(example['expected_result'], indent=6)}")
        print()
    
    print("=== Available Tools (34 total) ===")
    tool_categories = {
        "Navigation & Page Control": [
            "navigate(url: str) - Navigate to a URL",
            "reload() - Reload the current page", 
            "go_back() - Go back in history",
            "go_forward() - Go forward in history",
            "get_current_url() - Get current URL with parsed components and query parameters",
            "wait_for_url(url_pattern: str, timeout: int) - Wait for URL to match pattern",
            "wait_for_load_state(state: str, timeout: int) - Wait for page load states",
            "set_viewport_size(width: int, height: int) - Set viewport dimensions"
        ],
        "Element Interaction": [
            "click(selector: str) - Click an element",
            "type_text(selector: str, text: str) - Type text into an element", 
            "fill(selector: str, value: str) - Fill an input field",
            "clear_text(selector: str) - Clear input field text",
            "select_option(selector: str, value: str) - Select an option",
            "hover(selector: str) - Hover over an element",
            "scroll(selector: str, x: int, y: int) - Scroll element",
            "press_key(key: str) - Press keyboard key"
        ],
        "Form Handling": [
            "check_checkbox(selector: str) - Check a checkbox",
            "uncheck_checkbox(selector: str) - Uncheck a checkbox", 
            "upload_file(selector: str, file_path: str) - Upload file to input"
        ],
        "Element Discovery & Validation": [
            "query_selector(selector: str) - Query for single element",
            "query_selector_all(selector: str) - Query for all matching elements",
            "is_visible(selector: str) - Check if element is visible",
            "is_enabled(selector: str) - Check if element is enabled",
            "wait_for_element(selector: str, timeout: int) - Wait for element to appear",
            "get_element_bounding_box(selector: str) - Get element position and size",
            "get_element_attributes(selector: str) - Get all element attributes",
            "get_computed_style(selector: str, property: str) - Get CSS computed style"
        ],
        "Content & Snapshots": [
            "get_html() - Get page HTML",
            "get_accessibility_snapshot() - Get accessibility tree",
            "screenshot(selector: str, full_page: bool) - Take screenshot",
            "pdf() - Generate PDF of page"
        ],
        "JavaScript & Debugging": [
            "evaluate(script: str) - Execute JavaScript in page context",
            "wait_for_network_idle(timeout: int) - Wait for network activity to settle",
            "get_page_errors() - Get JavaScript errors from page",
            "get_console_logs() - Get console output from page"
        ]
    }
    
    for category, tools in tool_categories.items():
        print(f"\n{category}:")
        for tool in tools:
            print(f"  • {tool}")
    
    print("\n=== Usage Instructions ===")
    print("1. Install dependencies:")
    print("   pip install mcp playwright pydantic")
    print("   playwright install")
    print()
    print("2. Run the server:")
    print("   playwright-mcp stdio          # For MCP clients")
    print("   playwright-mcp http --port 8000  # For HTTP API")
    print()
    print("3. Connect from MCP clients:")
    print("   - Claude Desktop: Add to claude_desktop_config.json")
    print("   - MCP Inspector: uv run mcp dev src/playwright_mcp/server.py")
    print()
    print("4. Configuration options:")
    print("   --headed      Run browser in headed mode")
    print("   --browser     Choose browser: chromium, firefox, webkit")
    print("   --timeout     Set default timeout in milliseconds")

if __name__ == "__main__":
    asyncio.run(demo_browser_automation())