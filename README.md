# Playwright MCP Server

A minimal, robust Playwright MCP (Model Context Protocol) server that exposes core browser automation capabilities via a simple API.

## Features

- **Browser Context Management**: Persistent browser context (headless or headed, configurable)
- **Navigation**: Open URLs, reload, go back/forward
- **DOM Interaction**: Click, type, fill, select, hover, scroll using Playwright selectors
- **Element Discovery**: Query elements using CSS, XPath, role, text, and other Playwright locators
- **Snapshotting**: Get HTML, accessibility snapshots, screenshots, and PDFs
- **Script Evaluation**: Run JavaScript in the page context
- **Raw Output**: All outputs are raw Playwright results with no post-processing

## Installation

### Quick Install from GitHub

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/playwright-mcp.git

# Install Playwright browsers
playwright install
```

### For Development

```bash
# Clone the repository
git clone https://github.com/yourusername/playwright-mcp.git
cd playwright-mcp

# Install in development mode
pip install -e .

# Install browsers
playwright install
```

## Usage

### Running the Server

After installation, you can use it from anywhere:

```bash
# Run with stdio transport (for MCP clients)
playwright-mcp stdio

# Run with HTTP transport
playwright-mcp http --port 8000

# Run in headed mode (default is headless)
playwright-mcp stdio --headed
```

### Command Line Usage

```bash
# Run the MCP server
playwright-mcp stdio

# Run with visible browser
playwright-mcp stdio --headed

# Run HTTP server
playwright-mcp http --port 8000

# Use different browsers
playwright-mcp stdio --browser firefox
playwright-mcp stdio --browser webkit
```

### Integration with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "playwright": {
      "command": "playwright-mcp",
      "args": ["stdio"]
    }
  }
}
```

### Testing with MCP Inspector

```bash
# Install and run MCP inspector
uv run mcp dev src/playwright_mcp/server.py
```

## API Reference

### Tools

- `navigate(url: str)` - Navigate to a URL
- `reload()` - Reload the current page
- `go_back()` - Go back in history
- `go_forward()` - Go forward in history
- `click(selector: str)` - Click an element
- `type_text(selector: str, text: str)` - Type text into an element
- `fill(selector: str, value: str)` - Fill an input field
- `select_option(selector: str, value: str)` - Select an option
- `hover(selector: str)` - Hover over an element
- `scroll(selector: str, x: int, y: int)` - Scroll element
- `query_selector(selector: str)` - Query for element
- `query_selector_all(selector: str)` - Query for all matching elements
- `get_html()` - Get page HTML
- `get_accessibility_snapshot()` - Get accessibility tree
- `screenshot()` - Take screenshot
- `pdf()` - Generate PDF
- `evaluate(script: str)` - Run JavaScript

### Configuration

The server accepts the following configuration options:

- `--headed` / `--headless` - Run browser in headed or headless mode
- `--browser` - Browser type (chromium, firefox, webkit)
- `--port` - Port for HTTP transport
- `--timeout` - Default timeout for operations (ms)

## Development

```bash
# Clone the repository
git clone <repo-url>
cd playwright-mcp

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/
uv run ruff check src/

# Type check
uv run mypy src/
```

## License

MIT