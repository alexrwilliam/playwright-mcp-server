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
        ("get_current_url", "Get current page URL with parsed components and query parameters"),
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
        ("is_visible", "Check if element is visible"),
        ("is_enabled", "Check if element is enabled"),
        ("wait_for_element", "Wait for element to appear"),
        ("wait_for_load_state", "Wait for page load states"),
        ("clear_text", "Clear text from input field"),
        ("check_checkbox", "Check a checkbox"),
        ("uncheck_checkbox", "Uncheck a checkbox"),
        ("upload_file", "Upload file to input"),
        ("press_key", "Press keyboard key"),
        ("wait_for_url", "Wait for URL to match pattern"),
        ("set_viewport_size", "Set viewport dimensions"),
        ("get_element_bounding_box", "Get element position and size"),
        ("get_element_attributes", "Get all element attributes"),
        ("get_computed_style", "Get CSS computed style"),
        ("wait_for_network_idle", "Wait for network activity to settle"),
        ("get_page_errors", "Get JavaScript errors from page"),
        ("get_console_logs", "Get console output from page"),
        ("get_network_requests", "Get captured network requests"),
        ("get_network_responses", "Get captured network responses"),
        ("clear_network_logs", "Clear network monitoring logs"),
        ("intercept_route", "Intercept and handle network requests"),
        ("unroute_all", "Remove all route interceptors"),
        ("wait_for_response", "Wait for specific network response"),
        ("get_response_body", "Get response body for URL pattern"),
        ("get_cookies", "Retrieve cookies from browser"),
        ("add_cookies", "Add cookies to browser"),
        ("clear_cookies", "Clear cookies from browser"),
        ("get_local_storage", "Get localStorage data"),
        ("set_local_storage", "Set localStorage item"),
        ("get_session_storage", "Get sessionStorage data"),
        ("set_session_storage", "Set sessionStorage item"),
        ("clear_storage", "Clear localStorage and/or sessionStorage"),
        ("set_extra_headers", "Set additional HTTP headers"),
        ("set_user_agent", "Set User-Agent header"),
    ]
    
    print(f"📊 Total: {len(tools)} tools\n")
    
    categories = {
        "Navigation & URL": ["navigate", "reload", "go_back", "go_forward", "get_current_url", "wait_for_url"],
        "DOM Interaction": ["click", "type_text", "fill", "select_option", "hover", "scroll", "press_key"],
        "Form Handling": ["clear_text", "check_checkbox", "uncheck_checkbox", "upload_file"],
        "Element Discovery": ["query_selector", "query_selector_all", "is_visible", "is_enabled", "wait_for_element"],
        "Element Analysis": ["get_element_bounding_box", "get_element_attributes", "get_computed_style"],
        "Content Extraction": ["get_html", "get_accessibility_snapshot", "screenshot", "pdf"],
        "JavaScript & Debugging": ["evaluate", "get_page_errors", "get_console_logs"],
        "Network Monitoring": ["get_network_requests", "get_network_responses", "clear_network_logs", "wait_for_response", "get_response_body"],
        "Network Interception": ["intercept_route", "unroute_all"],
        "Page State": ["wait_for_load_state", "wait_for_network_idle", "set_viewport_size"],
        "Cookie Management": ["get_cookies", "add_cookies", "clear_cookies"],
        "Storage Management": ["get_local_storage", "set_local_storage", "get_session_storage", "set_session_storage", "clear_storage"],
        "Request Headers": ["set_extra_headers", "set_user_agent"]
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