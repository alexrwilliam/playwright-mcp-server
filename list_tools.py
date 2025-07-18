#!/usr/bin/env python3
"""List available tools in the Playwright MCP server."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def list_tools():
    """List all available tools following MCP patterns."""
    print("🛠️  Playwright MCP Server - Available Tools")
    print("=" * 50)
    
    tools = [
        ("navigate", "Navigate to a URL"),
        ("reload", "Reload the current page"),
        ("go_back", "Go back in browser history"),
        ("go_forward", "Go forward in browser history"),
        ("click", "Click an element using any Playwright selector"),
        ("type_text", "Type text into an element"),
        ("fill", "Fill an input field"),
        ("select_option", "Select an option from a dropdown"),
        ("hover", "Hover over an element"),
        ("scroll", "Scroll an element or the page"),
        ("query_selector", "Query for a single element"),
        ("query_selector_all", "Query for all matching elements"),
        ("get_html", "Get the full HTML of the current page"),
        ("get_accessibility_snapshot", "Get the accessibility tree snapshot"),
        ("screenshot", "Take a screenshot of the page or element"),
        ("pdf", "Generate a PDF of the current page"),
        ("evaluate", "Execute JavaScript in the page context"),
    ]
    
    print(f"📊 Total: {len(tools)} tools\n")
    
    categories = {
        "Navigation": ["navigate", "reload", "go_back", "go_forward"],
        "DOM Interaction": ["click", "type_text", "fill", "select_option", "hover", "scroll"],
        "Element Discovery": ["query_selector", "query_selector_all"],
        "Snapshotting": ["get_html", "get_accessibility_snapshot", "screenshot", "pdf"],
        "Script Execution": ["evaluate"]
    }
    
    for category, tool_names in categories.items():
        print(f"📋 {category}:")
        for tool_name, description in tools:
            if tool_name in tool_names:
                print(f"   • {tool_name:25} - {description}")
        print()
    
    print("🚀 Usage:")
    print("   playwright-mcp                 # Start MCP server (stdio)")
    print("   playwright-mcp --headed        # Start with visible browser")
    print("   uv run mcp dev playwright-mcp  # Test with MCP Inspector")

if __name__ == "__main__":
    list_tools()