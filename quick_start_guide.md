# Quick Start Guide - Playwright MCP Server

## 🚀 How to Get Started

### Step 1: Install Dependencies
```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install mcp playwright pydantic asyncio-throttle

# Install Playwright browsers
playwright install
```

### Step 2: Run the Server

#### Option A: For MCP Clients (recommended)
```bash
python3 src/playwright_mcp/server.py stdio
```

#### Option B: For HTTP API
```bash
python3 src/playwright_mcp/server.py http --port 8000
```

#### Option C: Run in Headed Mode (see browser)
```bash
python3 src/playwright_mcp/server.py stdio --headed
```

### Step 3: Test with MCP Inspector
```bash
# If you have uv installed
uv run mcp dev src/playwright_mcp/server.py

# This will open a web interface where you can test tools
```

### Step 4: Connect from Claude Desktop

Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "playwright": {
      "command": "python3",
      "args": [
        "/path/to/Playwright_MCP/src/playwright_mcp/server.py",
        "stdio"
      ]
    }
  }
}
```

## 🎯 What You Can Do

Once connected, you can ask Claude to:

1. **Navigate to websites**: "Navigate to https://example.com"
2. **Take screenshots**: "Take a screenshot of the current page"
3. **Find elements**: "Find all links on the page"
4. **Interact with elements**: "Click the login button"
5. **Fill forms**: "Fill the email field with test@example.com"
6. **Extract data**: "Get the HTML content of the page"
7. **Run JavaScript**: "Execute document.title in the browser"

## 🛠️ Available Tools

- `navigate(url)` - Go to a URL
- `reload()` - Refresh the page
- `go_back()` / `go_forward()` - Browser navigation
- `click(selector)` - Click elements
- `type_text(selector, text)` - Type text
- `fill(selector, value)` - Fill input fields
- `select_option(selector, value)` - Select dropdown options
- `hover(selector)` - Hover over elements
- `scroll(selector, x, y)` - Scroll elements
- `query_selector(selector)` - Find single element
- `query_selector_all(selector)` - Find all matching elements
- `get_html()` - Get page HTML
- `get_accessibility_snapshot()` - Get accessibility tree
- `screenshot(selector, full_page)` - Take screenshots
- `pdf()` - Generate PDF
- `evaluate(script)` - Run JavaScript

## 🔧 Configuration Options

- `--headed` - Run browser in headed mode (visible)
- `--browser chromium|firefox|webkit` - Choose browser
- `--timeout 30000` - Set timeout in milliseconds
- `--port 8000` - Set HTTP port

## 📝 Example Usage

```bash
# Run server for Claude Desktop
python3 src/playwright_mcp/server.py stdio

# Run server with visible browser
python3 src/playwright_mcp/server.py stdio --headed

# Run HTTP server on port 3000
python3 src/playwright_mcp/server.py http --port 3000

# Use Firefox instead of Chrome
python3 src/playwright_mcp/server.py stdio --browser firefox
```

## 🎪 Demo Without Installation

Want to see what the server looks like? Run:
```bash
python3 test_syntax.py
```

This will show you the server structure and all available tools without requiring any dependencies.

## 🆘 Troubleshooting

1. **Import errors**: Make sure you're in the virtual environment and all dependencies are installed
2. **Browser not found**: Run `playwright install` to download browsers
3. **Permission errors**: Make sure the server script is executable: `chmod +x src/playwright_mcp/server.py`
4. **Connection issues**: Check that the server is running and the path in claude_desktop_config.json is correct

## 🎁 What's Next?

Once you have the server running, you can:
- Ask Claude to automate web tasks
- Extract data from websites
- Take screenshots for documentation
- Test web applications
- Generate PDFs from web pages

The server provides raw Playwright functionality, so anything Playwright can do, your MCP clients can do!