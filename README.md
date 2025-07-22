# Playwright MCP Server

A minimal, robust Playwright MCP (Model Context Protocol) server that exposes core browser automation capabilities via a simple API.

## Features

- **Browser Context Management**: Persistent browser context (headless or headed, configurable)
- **Navigation**: Open URLs, reload, go back/forward
- **DOM Interaction**: Click, type, fill, select, hover, scroll using Playwright selectors
- **Element Discovery**: Query elements using CSS, XPath, role, text, and other Playwright locators
- **Snapshotting**: Get HTML, accessibility snapshots, screenshots, and PDFs
- **Script Evaluation**: Run JavaScript in the page context
- **Network Monitoring**: Capture and analyze all network requests and responses
- **Network Interception**: Block, modify, or mock network requests
- **Cookie Management**: Get, set, and clear browser cookies
- **Storage Access**: Manage localStorage and sessionStorage data
- **Headers & User Agent**: Customize request headers and browser identity
- **Raw Output**: All outputs are raw Playwright results with no post-processing

## Installation

### Quick Install from GitHub

```bash
# Install directly from GitHub
pip install git+https://github.com/alexrwilliam/playwright-mcp-server.git

# Install Playwright browsers
playwright install
```

### For Development

```bash
# Clone the repository
git clone https://github.com/alexrwilliam/playwright-mcp-server.git
cd playwright-mcp-server

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

#### Navigation & Page Control
- `navigate(url: str)` - Navigate to a URL
- `reload()` - Reload the current page
- `go_back()` - Go back in history
- `go_forward()` - Go forward in history
- `wait_for_url(url_pattern: str, timeout: int)` - Wait for URL to match pattern
- `wait_for_load_state(state: str, timeout: int)` - Wait for page load states (domcontentloaded, load, networkidle)
- `set_viewport_size(width: int, height: int)` - Set viewport dimensions

#### Element Interaction
- `click(selector: str)` - Click an element
- `type_text(selector: str, text: str)` - Type text into an element
- `fill(selector: str, value: str)` - Fill an input field
- `clear_text(selector: str)` - Clear input field text
- `select_option(selector: str, value: str)` - Select an option
- `hover(selector: str)` - Hover over an element
- `scroll(selector: str, x: int, y: int)` - Scroll element
- `press_key(key: str)` - Press keyboard key

#### Form Handling
- `check_checkbox(selector: str)` - Check a checkbox
- `uncheck_checkbox(selector: str)` - Uncheck a checkbox
- `upload_file(selector: str, file_path: str)` - Upload file to input

#### Element Discovery & Validation
- `query_selector(selector: str)` - Query for single element
- `query_selector_all(selector: str)` - Query for all matching elements
- `is_visible(selector: str)` - Check if element is visible
- `is_enabled(selector: str)` - Check if element is enabled
- `wait_for_element(selector: str, timeout: int)` - Wait for element to appear
- `get_element_bounding_box(selector: str)` - Get element position and size
- `get_element_attributes(selector: str)` - Get all element attributes
- `get_computed_style(selector: str, property: str)` - Get CSS computed style

#### Content & Snapshots
- `get_html()` - Get page HTML
- `get_accessibility_snapshot()` - Get accessibility tree
- `screenshot(selector: str, full_page: bool)` - Take screenshot of page or element
- `pdf()` - Generate PDF of page

#### JavaScript & Debugging
- `evaluate(script: str)` - Execute JavaScript in page context
- `wait_for_network_idle(timeout: int)` - Wait for network activity to settle
- `get_page_errors()` - Get JavaScript errors from page
- `get_console_logs()` - Get console output from page

#### Network Monitoring & Interception
- `get_network_requests(url_pattern: str)` - Retrieve captured network requests with filtering
- `get_network_responses(url_pattern: str)` - Retrieve captured network responses with filtering
- `clear_network_logs()` - Clear all captured network request/response logs
- `intercept_route(url_pattern: str, action: str, ...)` - Intercept and handle network requests
- `unroute_all()` - Remove all route interceptors
- `wait_for_response(url_pattern: str, timeout: int)` - Wait for specific network responses
- `get_response_body(url_pattern: str)` - Extract response body content from network calls

#### Cookie Management
- `get_cookies(urls: List[str])` - Retrieve browser cookies with optional URL filtering
- `add_cookies(cookies: List[Dict])` - Add cookies to browser context
- `clear_cookies(name: str, domain: str)` - Clear cookies with optional filtering

#### Storage Management
- `get_local_storage(origin: str)` - Access localStorage data
- `set_local_storage(key: str, value: str)` - Set localStorage items
- `get_session_storage()` - Access sessionStorage data
- `set_session_storage(key: str, value: str)` - Set sessionStorage items
- `clear_storage(storage_type: str)` - Clear localStorage and/or sessionStorage

#### Request Headers & Identity
- `set_extra_headers(headers: Dict)` - Add custom HTTP headers to all requests
- `set_user_agent(user_agent: str)` - Change browser User-Agent string

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