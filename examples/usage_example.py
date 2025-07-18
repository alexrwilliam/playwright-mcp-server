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
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   Tool: {example['tool']}")
        print(f"   Parameters: {json.dumps(example['params'], indent=6)}")
        print(f"   Expected Result: {json.dumps(example['expected_result'], indent=6)}")
        print()
    
    print("=== Available Tools ===")
    tools = [
        "navigate(url: str) - Navigate to a URL",
        "reload() - Reload the current page",
        "go_back() - Go back in history",
        "go_forward() - Go forward in history",
        "click(selector: str) - Click an element",
        "type_text(selector: str, text: str) - Type text into an element",
        "fill(selector: str, value: str) - Fill an input field",
        "select_option(selector: str, value: str) - Select an option",
        "hover(selector: str) - Hover over an element",
        "scroll(selector: str, x: int, y: int) - Scroll element",
        "query_selector(selector: str) - Query for element",
        "query_selector_all(selector: str) - Query for all matching elements",
        "get_html() - Get page HTML",
        "get_accessibility_snapshot() - Get accessibility tree",
        "screenshot(selector: str, full_page: bool) - Take screenshot",
        "pdf() - Generate PDF",
        "evaluate(script: str) - Run JavaScript"
    ]
    
    for tool in tools:
        print(f"  • {tool}")
    
    print("\n=== Usage Instructions ===")
    print("1. Install dependencies:")
    print("   pip install mcp playwright pydantic")
    print("   playwright install")
    print()
    print("2. Run the server:")
    print("   python -m playwright_mcp.server stdio          # For MCP clients")
    print("   python -m playwright_mcp.server http --port 8000  # For HTTP API")
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