"""Playwright MCP Server - Main server implementation."""

import asyncio
import os
import secrets
import base64
import shutil
import hashlib
import json
import logging
import mimetypes
import sys
import time
import uuid
import random

from playwright_mcp import cdp_patch
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse, parse_qs

from mcp.server.fastmcp import Context, FastMCP
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    Request,
    Response,
    Route,
)
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

MAX_STORED_RESPONSE_HANDLES = 500


@dataclass
class BrowserState:
    """Browser state container."""

    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page  # Current active page

    # Page management
    pages: Dict[str, Page] = None  # Maps page ID to Page object
    current_page_id: str = None  # ID of the current active page

    # Network monitoring state
    captured_requests: List[Dict[str, Any]] = None
    captured_responses: List[Dict[str, Any]] = None
    response_handles: Dict[str, Response] = None
    response_handle_order: Deque[str] = None

    def __post_init__(self):
        if self.pages is None:
            self.pages = {}
        if self.captured_requests is None:
            self.captured_requests = []
        if self.captured_responses is None:
            self.captured_responses = []
        if self.response_handles is None:
            self.response_handles = {}
        if self.response_handle_order is None:
            self.response_handle_order = deque()

    def get_current_page(self) -> Page:
        """Get the current active page."""
        if self.current_page_id and self.current_page_id in self.pages:
            return self.pages[self.current_page_id]
        return self.page

    def set_current_page(self, page_id: str):
        """Set the current active page by ID."""
        if page_id in self.pages:
            self.current_page_id = page_id
            self.page = self.pages[page_id]


class NavigationResult(BaseModel):
    """Navigation operation result."""

    success: bool
    url: str
    error: Optional[str] = None


class CurrentUrlResult(BaseModel):
    """Current URL information result."""

    success: bool
    url: Optional[str] = None
    parsed_url: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, List[str]]] = None
    error: Optional[str] = None


class ElementQueryResult(BaseModel):
    """Element query result."""

    found: bool
    count: int
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    returned_count: Optional[int] = None
    error: Optional[str] = None


class ElementMetaResult(BaseModel):
    """Lightweight element metadata result."""

    found: bool
    count: int
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    returned_count: Optional[int] = None
    error: Optional[str] = None


class ScreenshotResult(BaseModel):
    """Screenshot result."""

    success: bool
    data: Optional[str] = None  # base64 encoded
    format: str = "png"
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    byte_size: Optional[int] = None
    sha256: Optional[str] = None
    preview_base64: Optional[str] = None
    dimensions: Optional[Dict[str, Union[int, float]]] = None
    inline: bool = False


class PDFResult(BaseModel):
    """PDF generation result."""

    success: bool
    data: Optional[str] = None  # base64 encoded
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    byte_size: Optional[int] = None
    sha256: Optional[str] = None
    preview_base64: Optional[str] = None
    inline: bool = False


class ScriptResult(BaseModel):
    """JavaScript execution result."""

    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    truncated: Optional[bool] = None
    preview: Optional[str] = None
    overflow_path: Optional[str] = None
    overflow_characters: Optional[int] = None


class NetworkRequestResult(BaseModel):
    """Network request information result."""

    url: str
    method: str
    headers: Dict[str, str]
    resource_type: str
    post_data: Optional[str] = None
    timestamp: Optional[float] = None


class NetworkResponseResult(BaseModel):
    """Network response information result."""

    url: str
    status: int
    status_text: str
    headers: Dict[str, str]
    body: Optional[str] = None
    timestamp: Optional[float] = None


class CookieResult(BaseModel):
    """Cookie operation result."""

    success: bool
    cookies: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class StorageResult(BaseModel):
    """Storage operation result."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PageInfo(BaseModel):
    """Information about a browser page/tab."""

    page_id: str
    url: str
    title: str
    is_current: bool


class PageListResult(BaseModel):
    """Result of listing all open pages."""

    success: bool
    pages: List[PageInfo] = Field(default_factory=list)
    current_page_id: Optional[str] = None
    error: Optional[str] = None


class PageSwitchResult(BaseModel):
    """Result of switching between pages."""

    success: bool
    page_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


class NewPageResult(BaseModel):
    """Result of waiting for a new page to open."""

    success: bool
    page_id: Optional[str] = None
    url: Optional[str] = None
    opener_page_id: Optional[str] = None
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
        channel: Optional[str] = None,
        user_data_dir: Optional[str] = None,
        max_elements_returned: int = 20,
        max_element_text_length: int = 2000,
        max_accessibility_nodes: int = 500,
        max_response_characters: int = 4000,
        preview_characters: int = 400,
        artifact_directory: Optional[str] = None,
        artifact_max_age_seconds: int = 7200,
        artifact_max_files: int = 200,
        artifact_chunk_size: int = 4096,
        # Fingerprint and stealth controls
        user_agent: Optional[str] = None,
        accept_language: Optional[str] = None,
        languages: Optional[List[str]] = None,
        locale: Optional[str] = None,
        timezone_id: Optional[str] = None,
        device_scale_factor: Optional[float] = None,
        platform: Optional[str] = None,
        vendor: Optional[str] = None,
        hardware_concurrency: Optional[int] = None,
        device_memory: Optional[float] = None,
        sec_ch_ua: Optional[str] = None,
        sec_ch_ua_mobile: Optional[str] = None,
        sec_ch_ua_platform: Optional[str] = None,
        sec_ch_ua_platform_version: Optional[str] = None,
        sec_ch_ua_full_version_list: Optional[str] = None,
        stealth: bool = False,
        mask_devtools: bool = False,
        custom_init_script: Optional[str] = None,
        permission_overrides: Optional[Dict[str, str]] = None,
        grant_permissions: Optional[List[str]] = None,
        permissions_origin: Optional[str] = None,
        fill_plugins: bool = True,
        override_languages: bool = True,
        stub_chrome_runtime: bool = True,
        override_viewport: bool = True,
        force_devtools_closed: bool = True,
        apply_cdp_patch: bool = False,
    ):
        self.headless = headless
        self.browser_type = browser_type
        self.timeout = timeout
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.channel = channel
        self.user_data_dir = user_data_dir
        self.max_elements_returned = max_elements_returned
        self.max_element_text_length = max_element_text_length
        self.max_accessibility_nodes = max_accessibility_nodes
        self.max_response_characters = max_response_characters
        self.preview_characters = preview_characters
        # Base artifact directory; a per-session subdirectory is created at runtime.
        if artifact_directory:
            self.artifact_directory = Path(artifact_directory).expanduser()
        else:
            self.artifact_directory = (Path.cwd() / "tmp" / "playwright_mcp").expanduser()
        self.artifact_directory = self.artifact_directory.absolute()
        self.artifact_max_age_seconds = artifact_max_age_seconds
        self.artifact_max_files = artifact_max_files
        self.artifact_chunk_size = max(artifact_chunk_size, 256)
        # Fingerprinting / stealth defaults
        self.user_agent = user_agent
        self.accept_language = accept_language
        self.languages = languages or []
        self.locale = locale
        self.timezone_id = timezone_id
        self.device_scale_factor = device_scale_factor
        self.platform = platform
        self.vendor = vendor
        self.hardware_concurrency = hardware_concurrency
        self.device_memory = device_memory
        self.sec_ch_ua = sec_ch_ua
        self.sec_ch_ua_mobile = sec_ch_ua_mobile
        self.sec_ch_ua_platform = sec_ch_ua_platform
        self.sec_ch_ua_platform_version = sec_ch_ua_platform_version
        self.sec_ch_ua_full_version_list = sec_ch_ua_full_version_list
        self.stealth = stealth
        self.mask_devtools = mask_devtools
        self.custom_init_script = custom_init_script
        self.permission_overrides = permission_overrides or {}
        self.grant_permissions = grant_permissions or []
        self.permissions_origin = permissions_origin
        self.fill_plugins = fill_plugins
        self.override_languages = override_languages
        self.stub_chrome_runtime = stub_chrome_runtime
        self.override_viewport = override_viewport
        self.force_devtools_closed = force_devtools_closed
        self.apply_cdp_patch = apply_cdp_patch


# Global configuration
config = Config()
ANTIDETECT_PRESETS: Dict[str, Dict[str, Any]] = {
    "pixelscan": {
        "headless": False,
        "stealth": True,
        "channel": "chrome",
        # Keep webdriver masking but avoid devtools size spoofing which Pixelscan flags.
        "mask_devtools": False,
        # Let Chromium decide Accept-Language via locale to avoid mismatch warnings.
        "accept_language": None,
        # Let the browser-native language values flow through; overriding can look synthetic.
        "languages": [],
        "locale": "en-US",
        # Use native window/viewport; avoid forcing viewport values.
        "viewport_pool": [],
        # Avoid synthetic plugin/mime arrays; Pixelscan prefers native values.
        "fill_plugins": False,
        # Do not stub chrome.runtime; leave native chrome object untouched.
        "stub_chrome_runtime": False,
        # Do not override navigator.language/languages; use browser defaults.
        "override_languages": False,
        # Leave viewport untouched so outer/inner dimensions stay realistic.
        "override_viewport": False,
        # Explicitly report devtools closed without size spoofing.
        "force_devtools_closed": True,
        # Apply CDP transport patch to reduce control-channel fingerprints.
        "apply_cdp_patch": True,
    }
}
_restart_lock = asyncio.Lock()


def _get_antidetect_preset(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    return ANTIDETECT_PRESETS.get(name.lower())


def _apply_antidetect_preset(preset: Optional[Dict[str, Any]]):
    """Apply preset values to global config with support for viewport pooling."""

    if not preset:
        return

    viewport_pool = preset.get("viewport_pool") or []
    chosen_viewport: Optional[Dict[str, Any]] = None
    if viewport_pool:
        chosen_viewport = random.choice(viewport_pool)

    for key, value in preset.items():
        if key == "viewport_pool":
            continue
        setattr(config, key, value)

    if chosen_viewport and getattr(config, "override_viewport", True):
        config.viewport_width = chosen_viewport.get("width", config.viewport_width)
        config.viewport_height = chosen_viewport.get("height", config.viewport_height)
        if "device_scale_factor" in chosen_viewport:
            config.device_scale_factor = chosen_viewport["device_scale_factor"]
    if "apply_cdp_patch" in preset:
        config.apply_cdp_patch = preset["apply_cdp_patch"]


async def _shutdown_browser_state(state: Optional[BrowserState]):
    """Close browser/context and stop Playwright for a given state."""
    if not state:
        return

    try:
        if state.context:
            await state.context.close()
        elif state.browser:
            await state.browser.close()
    except Exception as exc:
        logger.error("Error closing browser: %s", exc)

    try:
        if state.playwright:
            await state.playwright.stop()
    except Exception as exc:
        logger.error("Error stopping playwright: %s", exc)


async def _prepare_context(context: BrowserContext):
    """Apply init scripts and permission grants to a freshly created context."""

    scripts = [_build_context_init_script(config), config.custom_init_script]
    for script in scripts:
        if not script:
            continue
        try:
            await context.add_init_script(script)
            for page in context.pages:
                try:
                    await page.add_init_script(script)
                except Exception as page_exc:
                    logger.debug("Failed to add init script to existing page: %s", page_exc)
        except Exception as exc:
            logger.warning("Failed to apply init script to context: %s", exc)

    if config.grant_permissions:
        try:
            await context.grant_permissions(config.grant_permissions, origin=config.permissions_origin)
        except Exception as exc:
            logger.warning("Failed to grant context permissions %s: %s", config.grant_permissions, exc)


async def _create_browser_state() -> BrowserState:
    """Launch Playwright and return an initialized BrowserState based on config."""
    if config.apply_cdp_patch:
        try:
            cdp_patch.apply()
        except Exception as exc:
            logger.warning("CDP patch apply failed: %s", exc)

    playwright = await async_playwright().start()

    use_persistent_context = config.user_data_dir is not None
    browser = None
    context = None

    context_options: Dict[str, Any] = {}
    if config.override_viewport:
        context_options["viewport"] = {
            "width": config.viewport_width,
            "height": config.viewport_height,
        }
        if config.device_scale_factor is not None:
            context_options["device_scale_factor"] = config.device_scale_factor
            context_options.setdefault(
                "screen",
                {"width": config.viewport_width, "height": config.viewport_height},
            )

    if config.user_agent:
        context_options["user_agent"] = config.user_agent

    locale_value = config.locale or (config.languages[0] if config.languages else None)
    if locale_value:
        context_options["locale"] = locale_value
    if config.timezone_id:
        context_options["timezone_id"] = config.timezone_id

    extra_headers = _build_extra_headers(config)
    if extra_headers:
        context_options["extra_http_headers"] = extra_headers

    try:
        if use_persistent_context:
            launch_options = {
                "headless": config.headless,
                **context_options,
            }
            if config.channel:
                launch_options["channel"] = config.channel
            if config.browser_type == "chromium" and config.stealth:
                launch_options["ignore_default_args"] = ["--enable-automation"]
                extra_args = launch_options.get("args", [])
                if "--disable-blink-features=AutomationControlled" not in extra_args:
                    extra_args.append("--disable-blink-features=AutomationControlled")
                launch_options["args"] = extra_args

            if config.browser_type == "chromium":
                context = await playwright.chromium.launch_persistent_context(
                    config.user_data_dir, **launch_options
                )
            elif config.browser_type == "firefox":
                context = await playwright.firefox.launch_persistent_context(
                    config.user_data_dir, **launch_options
                )
            elif config.browser_type == "webkit":
                context = await playwright.webkit.launch_persistent_context(
                    config.user_data_dir, **launch_options
                )
            else:
                raise ValueError(f"Unsupported browser type: {config.browser_type}")

            browser = context.browser
        else:
            launch_options = {"headless": config.headless}
            if config.channel:
                launch_options["channel"] = config.channel
            if config.browser_type == "chromium" and config.stealth:
                launch_options["ignore_default_args"] = ["--enable-automation"]
                extra_args = launch_options.get("args", [])
                if "--disable-blink-features=AutomationControlled" not in extra_args:
                    extra_args.append("--disable-blink-features=AutomationControlled")
                launch_options["args"] = extra_args

            if config.browser_type == "chromium":
                browser = await playwright.chromium.launch(**launch_options)
            elif config.browser_type == "firefox":
                browser = await playwright.firefox.launch(**launch_options)
            elif config.browser_type == "webkit":
                browser = await playwright.webkit.launch(**launch_options)
            else:
                raise ValueError(f"Unsupported browser type: {config.browser_type}")

            context = await browser.new_context(**context_options)

        await _prepare_context(context)

        page = context.pages[0] if context.pages else await context.new_page()

        page.set_default_timeout(config.timeout)

        state = BrowserState(
            playwright=playwright, browser=browser, context=context, page=page
        )

        page_id = str(uuid.uuid4())
        state.pages[page_id] = page
        state.current_page_id = page_id

        await _setup_page_tracking(state)
        await _setup_network_monitoring(state)

        logger.info("Browser started successfully")
        return state
    except Exception:
        # Best-effort cleanup on failure
        try:
            if context:
                await context.close()
            elif browser:
                await browser.close()
        except Exception:
            pass
        try:
            await playwright.stop()
        except Exception:
            pass
        raise


def _compute_session_id() -> str:
    """Compute a best-effort unique session identifier.

    We prefer terminal-provided session hints when available (to keep stability per tab),
    and add process/timestamp/randomness to guarantee uniqueness across concurrent runs.
    """
    hints: list[str] = []
    for var in (
        "MCP_SESSION_ID",  # explicit override if provided
        "CODEX_SESSION_ID",
        "ITERM_SESSION_ID",
        "TERM_SESSION_ID",
        "TMUX_PANE",
        "SSH_TTY",
    ):
        val = os.getenv(var)
        if val:
            hints.append(val)
    try:
        if sys.stdin and sys.stdin.isatty():
            # TTY device path is fairly unique per terminal tab
            hints.append(os.ttyname(sys.stdin.fileno()))
    except Exception:  # pragma: no cover - defensive
        pass

    # Always salt with pid + coarse time + small random to avoid collisions
    hints.extend([
        f"pid:{os.getpid()}",
        f"t:{int(time.time()*1000)}",
        f"r:{secrets.token_hex(4)}",
    ])

    basis = "|".join(hints)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _session_artifact_dir(base: Path) -> Path:
    """Return the per-session artifact directory under a given base."""
    sid = _compute_session_id()
    # Use a stable subfolder to group sessions, keeping base tidy
    return (base / "sessions" / sid).absolute()


def _enforce_session_retention(artifact_base: Path) -> None:
    """Prune old per-session artifact directories under the base root.

    Uses the same age limit as file retention (config.artifact_max_age_seconds).
    If set to 0 or negative, session pruning is disabled.
    """
    max_age = max(config.artifact_max_age_seconds, 0)
    if max_age <= 0:
        return

    sessions_root = (artifact_base / "sessions").expanduser().absolute()
    try:
        sessions_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Unable to ensure sessions root %s: %s", sessions_root, exc)
        return

    try:
        entries = [p for p in sessions_root.iterdir() if p.is_dir()]
    except Exception as exc:
        logger.warning("Failed to enumerate sessions in %s: %s", sessions_root, exc)
        return

    now = time.time()
    for path in entries:
        try:
            age = now - path.stat().st_mtime
            if age > max_age:
                shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            logger.debug("Failed to remove old session %s: %s", path, exc)


def _parse_csv_list(value: Optional[str]) -> List[str]:
    """Return a normalized list from a comma-separated string."""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_brand_list(header_value: Optional[str]) -> List[Dict[str, str]]:
    """Parse a Sec-CH-UA style header into a list of brand/version pairs."""

    if not header_value:
        return []

    brands: List[Dict[str, str]] = []
    for raw_part in header_value.split(","):
        part = raw_part.strip()
        if not part or ";v=" not in part:
            continue
        brand_part, version_part = part.split(";v=", 1)
        brand = brand_part.strip().strip('"')
        version = version_part.strip().strip('"')
        if brand:
            brands.append({"brand": brand, "version": version})
    return brands


def _parse_ch_mobile(value: Optional[str]) -> Optional[bool]:
    """Parse Sec-CH-UA-Mobile style values (?0/?1/true/false)."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"?1", "1", "true", "yes"}:
        return True
    if normalized in {"?0", "0", "false", "no"}:
        return False
    return None


def _build_client_hints_payload(cfg: Config) -> Dict[str, Any]:
    """Build structured client hints data from config."""

    brands = _parse_brand_list(cfg.sec_ch_ua)
    full_version_list = _parse_brand_list(cfg.sec_ch_ua_full_version_list)
    mobile = _parse_ch_mobile(cfg.sec_ch_ua_mobile)

    payload: Dict[str, Any] = {}
    if brands:
        payload["brands"] = brands
    if full_version_list:
        payload["fullVersionList"] = full_version_list
    if mobile is not None:
        payload["mobile"] = mobile
    if cfg.sec_ch_ua_platform:
        payload["platform"] = cfg.sec_ch_ua_platform
    if cfg.sec_ch_ua_platform_version:
        payload["platformVersion"] = cfg.sec_ch_ua_platform_version
    return payload


def _build_extra_headers(cfg: Config) -> Dict[str, str]:
    """Construct coherent headers (UA + client hints + locale) for the context."""

    headers: Dict[str, str] = {}
    if cfg.user_agent:
        headers["User-Agent"] = cfg.user_agent

    languages = cfg.languages or []
    if cfg.accept_language:
        headers["Accept-Language"] = cfg.accept_language
    elif languages:
        headers["Accept-Language"] = ",".join(languages)

    if cfg.sec_ch_ua:
        headers["Sec-CH-UA"] = cfg.sec_ch_ua
    if cfg.sec_ch_ua_full_version_list:
        headers["Sec-CH-UA-Full-Version-List"] = cfg.sec_ch_ua_full_version_list
    if cfg.sec_ch_ua_mobile:
        headers["Sec-CH-UA-Mobile"] = cfg.sec_ch_ua_mobile
    if cfg.sec_ch_ua_platform:
        headers["Sec-CH-UA-Platform"] = cfg.sec_ch_ua_platform
    if cfg.sec_ch_ua_platform_version:
        headers["Sec-CH-UA-Platform-Version"] = cfg.sec_ch_ua_platform_version

    return headers


def _build_context_init_script(cfg: Config) -> Optional[str]:
    """Build a single init script that applies stealth and fingerprint settings early."""

    active = any(
        [
            cfg.stealth,
            cfg.user_agent,
            cfg.languages,
            cfg.accept_language,
            cfg.platform,
            cfg.vendor,
            cfg.hardware_concurrency,
            cfg.device_memory,
            cfg.device_scale_factor,
            cfg.permission_overrides,
            cfg.sec_ch_ua
            or cfg.sec_ch_ua_full_version_list
            or cfg.sec_ch_ua_mobile
            or cfg.sec_ch_ua_platform
            or cfg.sec_ch_ua_platform_version,
        ]
    )
    if not active:
        return None

    options: Dict[str, Any] = {
        "userAgent": cfg.user_agent,
        "languages": cfg.languages,
        "acceptLanguage": cfg.accept_language,
        "platform": cfg.platform,
        "vendor": cfg.vendor,
        "hardwareConcurrency": cfg.hardware_concurrency,
        "deviceMemory": cfg.device_memory,
        "deviceScaleFactor": cfg.device_scale_factor,
        "viewport": {
            "width": cfg.viewport_width,
            "height": cfg.viewport_height,
        } if cfg.override_viewport else None,
        "clientHints": _build_client_hints_payload(cfg) or None,
        "stealth": cfg.stealth,
        "maskDevtools": cfg.mask_devtools,
        "fillPlugins": cfg.fill_plugins,
        "permissionStates": cfg.permission_overrides,
        "overrideLanguages": cfg.override_languages,
        "stubChromeRuntime": cfg.stub_chrome_runtime,
        "overrideViewport": cfg.override_viewport,
        "forceDevtoolsClosed": cfg.force_devtools_closed,
    }

    script = f"""
(() => {{
  const cfg = {json.dumps(options)};

  const defineValue = (target, key, value) => {{
    try {{
      Object.defineProperty(target, key, {{
        get: () => value,
        configurable: true,
      }});
    }} catch (_err) {{}}
  }};

  const defineGetter = (target, key, getter) => {{
    try {{
      Object.defineProperty(target, key, {{
        get: getter,
        configurable: true,
      }});
    }} catch (_err) {{}}
  }};

  // Prefer removing webdriver entirely so `'webdriver' in navigator` is false.
  try {{ delete Navigator.prototype.webdriver; }} catch (_err) {{}}
  try {{ delete navigator.webdriver; }} catch (_err) {{}}

  if (cfg.userAgent) {{
    defineValue(navigator, 'userAgent', cfg.userAgent);
    defineValue(navigator, 'appVersion', cfg.userAgent);
  }}
  if (cfg.platform) defineValue(navigator, 'platform', cfg.platform);
  if (cfg.vendor) defineValue(navigator, 'vendor', cfg.vendor);
  if (cfg.hardwareConcurrency) defineValue(navigator, 'hardwareConcurrency', cfg.hardwareConcurrency);
  if (cfg.deviceMemory) defineValue(navigator, 'deviceMemory', cfg.deviceMemory);

  if (cfg.overrideLanguages) {{
    const langs = Array.isArray(cfg.languages) ? cfg.languages.filter(Boolean) : [];
    if (langs.length) {{
      defineValue(navigator, 'languages', langs.slice());
      defineValue(navigator, 'language', langs[0]);
    }} else if (cfg.acceptLanguage) {{
      const pieces = cfg.acceptLanguage.split(',').map(p => p.trim()).filter(Boolean);
      if (pieces.length) {{
        defineValue(navigator, 'languages', pieces);
        defineValue(navigator, 'language', pieces[0]);
      }}
    }}
  }}

  if (cfg.deviceScaleFactor) {{
    defineValue(window, 'devicePixelRatio', cfg.deviceScaleFactor);
  }}

  if (cfg.overrideViewport && cfg.viewport && cfg.viewport.width && cfg.viewport.height) {{
    const width = cfg.viewport.width;
    const height = cfg.viewport.height;
    const screenObj = window.screen || {{}};
    defineValue(screenObj, 'width', width);
    defineValue(screenObj, 'height', height);
    defineValue(screenObj, 'availWidth', width);
    defineValue(screenObj, 'availHeight', height);
    try {{
      if (!window.screen) {{
        Object.defineProperty(window, 'screen', {{ get: () => screenObj, configurable: true }});
      }}
    }} catch (_err) {{}}
  }}

  const permissionStates = cfg.permissionStates || {{}};
  if (navigator.permissions && navigator.permissions.query) {{
    const originalQuery = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (parameters) => {{
      const name = parameters && parameters.name;
      if (name && permissionStates[name]) {{
        return Promise.resolve({{ state: permissionStates[name], onchange: null }});
      }}
      return originalQuery(parameters);
    }};
  }}

  if (cfg.clientHints) {{
    const ch = cfg.clientHints;
    const uaData = {{
      get brands() {{ return ch.brands || []; }},
      get mobile() {{ return !!ch.mobile; }},
      get platform() {{ return ch.platform || ''; }},
      get platformVersion() {{ return ch.platformVersion || ''; }},
      get architecture() {{ return ch.architecture || ''; }},
      get model() {{ return ch.model || ''; }},
      get uaFullVersion() {{
        if (ch.fullVersion) return ch.fullVersion;
        if (Array.isArray(ch.fullVersionList) && ch.fullVersionList.length) {{
          return ch.fullVersionList[0].version || '';
        }}
        return '';
      }},
      get fullVersionList() {{ return ch.fullVersionList || []; }},
      toJSON() {{ return {{
        brands: this.brands,
        mobile: this.mobile,
        platform: this.platform,
        platformVersion: this.platformVersion,
        fullVersionList: this.fullVersionList,
      }}; }},
    }};
    defineValue(navigator, 'userAgentData', uaData);
  }}

  if (cfg.forceDevtoolsClosed) {{
    ['__IS_DEVTOOLS_OPEN__', '__isDevtoolsOpen', '__devtoolsOpen', 'isDevtoolsOpen', 'isDevtoolOpen'].forEach((key) => {{
      try {{ defineValue(window, key, false); }} catch (_err) {{}}
    }});
  }}

  if (cfg.stealth) {{
    if (cfg.stubChromeRuntime) {{
      if (!window.chrome) {{
        window.chrome = {{ runtime: {{ connect: () => ({{}}), sendMessage: () => undefined, onMessage: {{ addListener: () => undefined, removeListener: () => undefined }} }} }};
      }} else if (!window.chrome.runtime) {{
        window.chrome.runtime = {{ connect: () => ({{}}), sendMessage: () => undefined, onMessage: {{ addListener: () => undefined, removeListener: () => undefined }} }};
      }}
    }}

    if (cfg.fillPlugins) {{
      const pluginEntries = [
        {{ name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' }},
        {{ name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' }},
        {{ name: 'Native Client', filename: 'internal-nacl-plugin', description: '' }},
      ];
      const mimeTypes = [
        {{ type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format', enabledPlugin: undefined }},
        {{ type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Portable Document Format', enabledPlugin: undefined }},
        {{ type: 'application/x-nacl', suffixes: '', description: 'Native Client Executable', enabledPlugin: undefined }},
        {{ type: 'application/x-pnacl', suffixes: '', description: 'Portable Native Client Executable', enabledPlugin: undefined }},
      ];
      if (!navigator.plugins || navigator.plugins.length === 0) {{
        const pluginArray = pluginEntries.map((p) => Object.assign({{ 0: mimeTypes[0], length: 1 }}, p));
        defineValue(navigator, 'plugins', pluginArray);
      }}
      if (!navigator.mimeTypes || navigator.mimeTypes.length === 0) {{
        defineValue(navigator, 'mimeTypes', mimeTypes);
      }}
    }}
    try {{ delete window.cdc_adoQpoasnfa76pfcZLmcfl_; }} catch (_err) {{}}
    try {{ delete window.__nightmare; }} catch (_err) {{}}
  }}

  if (cfg.maskDevtools) {{
    const fallbackGap = 120;
    defineGetter(window, 'outerWidth', () => {{
      const base = window.innerWidth || 0;
      const avail = (window.screen && window.screen.availWidth) || base;
      return Math.max(base + fallbackGap, avail);
    }});
    defineGetter(window, 'outerHeight', () => {{
      const base = window.innerHeight || 0;
      const avail = (window.screen && window.screen.availHeight) || base;
      return Math.max(base + fallbackGap, avail);
    }});
    ['__IS_DEVTOOLS_OPEN__', '__isDevtoolsOpen', '__devtoolsOpen'].forEach((key) => defineValue(window, key, false));
    if (!window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {{
      defineValue(window, '__REACT_DEVTOOLS_GLOBAL_HOOK__', undefined);
    }}
  }}
}})();
    """

    return script


@asynccontextmanager
async def browser_lifespan(server: FastMCP) -> AsyncIterator[BrowserState]:
    """Manage browser lifecycle."""
    logger.info("Starting browser...")
    state: Optional[BrowserState] = None
    try:
        state = await _create_browser_state()
        yield state
    finally:
        logger.info("Shutting down browser...")
        await _shutdown_browser_state(state)


# Create FastMCP server with browser lifespan
mcp = FastMCP("Playwright MCP Server", lifespan=browser_lifespan)


async def _setup_page_tracking(state: BrowserState):
    """Set up page tracking for new tabs and popups."""
    async def handle_new_page(page: Page):
        """Handle new page/tab creation."""
        try:
            # Generate unique ID for the new page
            page_id = str(uuid.uuid4())

            # Set default timeout for new page
            page.set_default_timeout(config.timeout)

            # Add to pages dictionary
            state.pages[page_id] = page

            # Set up network monitoring for the new page
            await _setup_network_monitoring_for_page(state, page)

            logger.info(f"New page opened with ID: {page_id}, URL: {page.url}")
        except Exception as e:
            logger.error(f"Error handling new page: {e}")

    # Listen for new pages (tabs, popups, etc.)
    state.context.on("page", handle_new_page)


async def _setup_network_monitoring_for_page(state: BrowserState, page: Page):
    """Set up network monitoring for a specific page."""
    async def handle_request(request: Request):
        """Capture request details."""
        try:
            request_data = {
                "url": request.url,
                "method": request.method,
                "headers": await request.all_headers(),
                "resource_type": request.resource_type,
                "post_data": request.post_data,
                "timestamp": time.time(),
            }
            state.captured_requests.append(request_data)
        except Exception as e:
            logger.error(f"Error capturing request: {e}")

    async def handle_response(response: Response):
        """Capture response details."""
        try:
            handle_id = uuid.uuid4().hex
            state.response_handles[handle_id] = response
            state.response_handle_order.append(handle_id)
            while len(state.response_handle_order) > MAX_STORED_RESPONSE_HANDLES:
                old_id = state.response_handle_order.popleft()
                state.response_handles.pop(old_id, None)

            response_data = {
                "url": response.url,
                "status": response.status,
                "status_text": response.status_text,
                "headers": await response.all_headers(),
                "timestamp": time.time(),
                "handle_id": handle_id,
            }
            state.captured_responses.append(response_data)
        except Exception as e:
            logger.error(f"Error capturing response: {e}")

    # Set up event listeners for this page
    page.on("request", handle_request)
    page.on("response", handle_response)


async def _setup_network_monitoring(state: BrowserState):
    """Set up network request and response monitoring for initial page."""
    await _setup_network_monitoring_for_page(state, state.page)


def get_browser_state(ctx: Context) -> BrowserState:
    """Get browser state from context."""
    return ctx.request_context.lifespan_context


def get_current_page(ctx: Context) -> Page:
    """Get the current active page from context."""
    browser_state = get_browser_state(ctx)
    return browser_state.get_current_page()


def _truncate_text(value: Optional[str], max_length: int) -> Tuple[Optional[str], bool]:
    """Truncate string values according to configured limit."""
    if value is None:
        return None, False
    if max_length <= 0 or len(value) <= max_length:
        return value, False
    return value[:max_length], True


def _truncate_attributes(
    attributes: Dict[str, Any], max_length: int
) -> Tuple[Dict[str, Any], bool]:
    """Apply text truncation to string attribute values."""
    truncated = False
    limited: Dict[str, Any] = {}
    for key, attr_value in attributes.items():
        if isinstance(attr_value, str):
            limited_value, was_truncated = _truncate_text(attr_value, max_length)
            truncated = truncated or was_truncated
            limited[key] = limited_value
        else:
            limited[key] = attr_value
    return limited, truncated


def _count_accessibility_nodes(node: Optional[Dict[str, Any]]) -> int:
    if not node:
        return 0
    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        total += 1
        children = current.get("children") or []
        stack.extend(children)
    return total


def _prune_accessibility_snapshot(
    snapshot: Optional[Dict[str, Any]], max_nodes: int
) -> Tuple[Optional[Dict[str, Any]], bool, int]:
    if snapshot is None:
        return None, False, 0
    if max_nodes <= 0:
        return snapshot, False, _count_accessibility_nodes(snapshot)

    included = 0
    truncated = False

    def prune(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal included, truncated
        if included >= max_nodes:
            truncated = True
            return None

        node_copy = dict(node)
        included += 1

        children = node_copy.get("children") or []
        if children:
            pruned_children = []
            for child in children:
                if included >= max_nodes:
                    truncated = True
                    break
                pruned_child = prune(child)
                if pruned_child is not None:
                    pruned_children.append(pruned_child)
            if pruned_children:
                node_copy["children"] = pruned_children
            else:
                node_copy.pop("children", None)

        return node_copy

    pruned_snapshot = prune(snapshot)
    return pruned_snapshot, truncated, included


def _resolve_limit(default_value: int, override: Optional[int]) -> int:
    if override is None:
        return default_value
    return override


def _ensure_artifact_dir() -> Path:
    """Ensure the artifact directory exists and return it."""
    directory = config.artifact_directory
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Unable to create artifact directory %s: %s", directory, exc)
    return directory


def _make_artifact_path(label: str, suffix: str = ".json") -> Path:
    """Create a unique artifact file path for overflow payloads."""
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.lower())
    safe_label = safe_label[:64] or "payload"
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename = f"{timestamp}_{safe_label}_{uuid.uuid4().hex}{suffix}"
    return _ensure_artifact_dir() / filename


def _resolve_artifact_path(path: str) -> Path:
    """Resolve a path inside the artifact directory, ensuring it is safe."""

    try:
        resolved_path = Path(path).expanduser().absolute()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid path: {exc}") from exc

    artifact_root = _ensure_artifact_dir().absolute()
    try:
        resolved_path.relative_to(artifact_root)
    except ValueError as exc:
        raise PermissionError("Path is outside of the configured artifact directory") from exc

    if not resolved_path.exists():
        raise FileNotFoundError("Artifact not found")

    return resolved_path


def _is_probably_binary(data: bytes) -> bool:
    """Heuristic check for binary data."""

    if not data:
        return False
    text_chars = {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F))
    control_bytes = sum(byte not in text_chars for byte in data)
    return control_bytes / max(len(data), 1) > 0.30


def _guess_suffix_from_mime(content_type: str, default: str) -> str:
    """Return a file suffix based on MIME type."""

    if not content_type:
        return default
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
    if guessed:
        return guessed
    if content_type.startswith("image/"):
        return ".png"
    if content_type.startswith("application/pdf"):
        return ".pdf"
    if content_type.startswith("text/") or content_type.endswith("json"):
        return ".txt"
    return default


def _is_text_content(content_type: str) -> bool:
    """Determine whether a MIME type represents text content."""

    if not content_type:
        return True

    ctype = content_type.split(";")[0].strip().lower()
    if ctype.startswith("text/"):
        return True
    if ctype.endswith("json") or ctype.endswith("+json"):
        return True
    if ctype.endswith("xml") or ctype.endswith("+xml"):
        return True
    return ctype in {
        "application/javascript",
        "application/x-javascript",
        "application/vnd.mozilla.xul+xml",
    }


def _serialize_preview(value: Any, max_chars: int) -> str:
    """Return a string preview of a value limited to ``max_chars``."""
    if max_chars <= 0:
        return ""
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        serialized = str(value)
    if len(serialized) <= max_chars:
        return serialized
    return serialized[:max_chars]


def _estimate_size(value: Any) -> int:
    """Estimate the serialized size of the value in characters."""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def _empty_like(value: Any) -> Any:
    """Return an empty placeholder of the same container type."""
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    if isinstance(value, str):
        return ""
    return None


def _shrink_value(value: Any, budget: int) -> Tuple[Any, bool]:
    """Shrink the value so that its serialized representation fits within ``budget`` characters."""
    if budget <= 0:
        return _empty_like(value), True

    if isinstance(value, str):
        if len(value) <= budget:
            return value, False
        return value[:budget], True

    if isinstance(value, list):
        truncated = False
        reduced: List[Any] = []
        for item in value:
            current_size = _estimate_size(reduced)
            remaining = max(budget - current_size - 1, 0)
            if remaining <= 0:
                truncated = True
                break
            shrunk_item, item_truncated = _shrink_value(item, remaining)
            candidate = reduced + [shrunk_item]
            if _estimate_size(candidate) > budget:
                truncated = True
                break
            reduced.append(shrunk_item)
            truncated = truncated or item_truncated
        return reduced, truncated

    if isinstance(value, dict):
        truncated = False
        reduced: Dict[Any, Any] = {}
        for key, item in value.items():
            current_size = _estimate_size(reduced)
            key_overhead = _estimate_size({key: None}) - _estimate_size({})
            remaining = max(budget - current_size - key_overhead, 0)
            if remaining <= 0:
                truncated = True
                break
            shrunk_item, item_truncated = _shrink_value(item, remaining)
            candidate = dict(reduced)
            candidate[key] = shrunk_item
            if _estimate_size(candidate) > budget:
                truncated = True
                break
            reduced[key] = shrunk_item
            truncated = truncated or item_truncated
        return reduced, truncated

    return value, False


def _apply_response_budget(
    value: Any,
    *,
    budget: int,
    preview_limit: int,
    label: str,
) -> Tuple[Any, bool, Optional[str], Optional[str], int]:
    """Apply response budget constraints to the provided value."""

    size = _estimate_size(value)
    if budget <= 0 or size <= budget:
        return value, False, None, None, size

    trimmed_value, trimmed_internally = _shrink_value(value, budget)
    # Ensure the trimmed value actually respects the budget; fallback to empty if not.
    if _estimate_size(trimmed_value) > budget:
        trimmed_value = _empty_like(value)
        trimmed_internally = True

    preview = _serialize_preview(trimmed_value, preview_limit)
    artifact_path = _make_artifact_path(label)
    serialized_value = None
    try:
        serialized_value = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        serialized_value = str(value)
        artifact_path = artifact_path.with_suffix(".txt")

    try:
        artifact_path.write_text(serialized_value, encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to write overflow artifact %s: %s", artifact_path, exc)
        artifact_path = None

    if artifact_path:
        _enforce_artifact_retention()

    return trimmed_value, True, preview, str(artifact_path) if artifact_path else None, size


def _enforce_artifact_retention() -> None:
    """Apply retention policy to the artifact directory."""

    directory = config.artifact_directory
    max_age = max(config.artifact_max_age_seconds, 0)
    max_files = config.artifact_max_files

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Unable to ensure artifact directory %s: %s", directory, exc)
        return

    try:
        files = [path for path in directory.iterdir() if path.is_file()]
    except Exception as exc:
        logger.error("Failed to enumerate artifact directory %s: %s", directory, exc)
        return

    now = time.time()

    # Remove files older than the configured age limit
    if max_age > 0:
        for path in list(files):
            try:
                if now - path.stat().st_mtime > max_age:
                    path.unlink(missing_ok=True)
                    files.remove(path)
            except Exception as exc:
                logger.warning("Failed to remove expired artifact %s: %s", path, exc)

    # Enforce maximum file count by removing oldest files first
    if max_files and max_files > 0 and len(files) > max_files:
        try:
            files.sort(key=lambda p: p.stat().st_mtime)
        except Exception as exc:
            logger.error("Failed to sort artifacts for retention: %s", exc)
            return

        for path in files[:-max_files]:
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to remove excess artifact %s: %s", path, exc)

META_ATTRIBUTE_KEYS = {
    "id",
    "class",
    "name",
    "role",
    "aria-label",
    "aria-labelledby",
    "aria-describedby",
    "data-testid",
    "data-test",
    "data-qa",
}


# Browser lifecycle tools
@mcp.tool()
async def restart_browser(ctx: Context, headed: bool = True) -> Dict[str, Any]:
    """Restart the browser in headed or headless mode."""
    async with _restart_lock:
        state = get_browser_state(ctx)
        previous_headless = config.headless
        config.headless = not headed

        try:
            new_state = await _create_browser_state()
        except Exception as exc:
            config.headless = previous_headless
            logger.error("Failed to start new browser during restart: %s", exc)
            return {"success": False, "headed": headed, "error": str(exc)}

        try:
            await _shutdown_browser_state(state)
        except Exception as exc:
            logger.warning("Error shutting down previous browser during restart: %s", exc)

        state.playwright = new_state.playwright
        state.browser = new_state.browser
        state.context = new_state.context
        state.page = new_state.page
        state.pages = new_state.pages
        state.current_page_id = new_state.current_page_id
        state.captured_requests = new_state.captured_requests
        state.captured_responses = new_state.captured_responses
        state.response_handles = new_state.response_handles
        state.response_handle_order = new_state.response_handle_order

        return {
            "success": True,
            "headed": headed,
            "current_page_id": state.current_page_id,
        }


@mcp.tool()
async def recreate_context(
    ctx: Context,
    antidetect_preset: Optional[str] = None,
    headless: Optional[bool] = None,
    stealth: Optional[bool] = None,
    mask_devtools: Optional[bool] = None,
    user_agent: Optional[str] = None,
    accept_language: Optional[str] = None,
    languages: Optional[List[str]] = None,
    locale: Optional[str] = None,
    timezone_id: Optional[str] = None,
    platform: Optional[str] = None,
    vendor: Optional[str] = None,
    hardware_concurrency: Optional[int] = None,
    device_memory: Optional[float] = None,
    device_scale_factor: Optional[float] = None,
    sec_ch_ua: Optional[str] = None,
    sec_ch_ua_mobile: Optional[str] = None,
    sec_ch_ua_platform: Optional[str] = None,
    sec_ch_ua_platform_version: Optional[str] = None,
    sec_ch_ua_full_version_list: Optional[str] = None,
    init_script: Optional[str] = None,
    clear_init_script: bool = False,
    grant_permissions: Optional[List[str]] = None,
    permissions_origin: Optional[str] = None,
    permission_overrides: Optional[Dict[str, str]] = None,
    viewport_width: Optional[int] = None,
    viewport_height: Optional[int] = None,
) -> Dict[str, Any]:
    """Recreate the browser context with new fingerprint/stealth/init-script settings.

    This closes the existing context and spins up a new one with the provided
    overrides applied at creation time (UA, client hints, languages/locale/timezone,
    device metrics, permissions, stealth/devtools masking, and init scripts).
    """
    async with _restart_lock:
        state = get_browser_state(ctx)

        # Snapshot current config so we can roll back on failure.
        previous = {
            "headless": config.headless,
            "stealth": config.stealth,
            "mask_devtools": config.mask_devtools,
            "user_agent": config.user_agent,
            "accept_language": config.accept_language,
            "languages": list(config.languages),
            "locale": config.locale,
            "timezone_id": config.timezone_id,
            "platform": config.platform,
            "vendor": config.vendor,
            "hardware_concurrency": config.hardware_concurrency,
            "device_memory": config.device_memory,
            "device_scale_factor": config.device_scale_factor,
            "sec_ch_ua": config.sec_ch_ua,
            "sec_ch_ua_mobile": config.sec_ch_ua_mobile,
            "sec_ch_ua_platform": config.sec_ch_ua_platform,
            "sec_ch_ua_platform_version": config.sec_ch_ua_platform_version,
            "sec_ch_ua_full_version_list": config.sec_ch_ua_full_version_list,
            "custom_init_script": config.custom_init_script,
            "grant_permissions": list(config.grant_permissions),
            "permission_overrides": dict(config.permission_overrides),
            "permissions_origin": config.permissions_origin,
            "viewport_width": config.viewport_width,
            "viewport_height": config.viewport_height,
        }

        preset = _get_antidetect_preset(antidetect_preset)
        if antidetect_preset and not preset:
            return {"success": False, "error": f"Unknown anti-detect preset: {antidetect_preset}"}
        _apply_antidetect_preset(preset)

        def set_if(value: Optional[Any], attr: str):
            if value is not None:
                setattr(config, attr, value)

        set_if(headless, "headless")
        set_if(stealth, "stealth")
        if mask_devtools is not None:
            config.mask_devtools = mask_devtools
        elif stealth is not None and config.mask_devtools is None:
            config.mask_devtools = stealth

        set_if(user_agent, "user_agent")
        set_if(accept_language, "accept_language")
        if languages is not None:
            config.languages = languages
        set_if(locale, "locale")
        set_if(timezone_id, "timezone_id")
        set_if(platform, "platform")
        set_if(vendor, "vendor")
        set_if(hardware_concurrency, "hardware_concurrency")
        set_if(device_memory, "device_memory")
        set_if(device_scale_factor, "device_scale_factor")
        set_if(sec_ch_ua, "sec_ch_ua")
        set_if(sec_ch_ua_mobile, "sec_ch_ua_mobile")
        set_if(sec_ch_ua_platform, "sec_ch_ua_platform")
        set_if(sec_ch_ua_platform_version, "sec_ch_ua_platform_version")
        set_if(sec_ch_ua_full_version_list, "sec_ch_ua_full_version_list")
        if clear_init_script:
            config.custom_init_script = None
        if init_script is not None:
            config.custom_init_script = init_script
        if grant_permissions is not None:
            config.grant_permissions = grant_permissions
        if permission_overrides is not None:
            config.permission_overrides = permission_overrides
        if permissions_origin is not None:
            config.permissions_origin = permissions_origin
        if viewport_width is not None:
            config.viewport_width = viewport_width
        if viewport_height is not None:
            config.viewport_height = viewport_height

        try:
            new_state = await _create_browser_state()
        except Exception as exc:
            # Roll back on failure.
            for key, value in previous.items():
                setattr(config, key, value)
            logger.error("Failed to recreate context with new settings: %s", exc)
            return {"success": False, "error": str(exc)}

        try:
            await _shutdown_browser_state(state)
        except Exception as exc:
            logger.warning("Error shutting down previous browser during recreate: %s", exc)

        state.playwright = new_state.playwright
        state.browser = new_state.browser
        state.context = new_state.context
        state.page = new_state.page
        state.pages = new_state.pages
        state.current_page_id = new_state.current_page_id
        state.captured_requests = new_state.captured_requests
        state.captured_responses = new_state.captured_responses
        state.response_handles = new_state.response_handles
        state.response_handle_order = new_state.response_handle_order

        applied = {
            "stealth": config.stealth,
            "mask_devtools": config.mask_devtools,
            "user_agent": config.user_agent,
            "accept_language": config.accept_language,
            "languages": config.languages,
            "locale": config.locale,
            "timezone_id": config.timezone_id,
            "platform": config.platform,
            "vendor": config.vendor,
            "hardware_concurrency": config.hardware_concurrency,
            "device_memory": config.device_memory,
            "device_scale_factor": config.device_scale_factor,
            "sec_ch_ua": config.sec_ch_ua,
            "sec_ch_ua_mobile": config.sec_ch_ua_mobile,
            "sec_ch_ua_platform": config.sec_ch_ua_platform,
            "sec_ch_ua_platform_version": config.sec_ch_ua_platform_version,
            "sec_ch_ua_full_version_list": config.sec_ch_ua_full_version_list,
            "permissions_origin": config.permissions_origin,
            "granted_permissions": config.grant_permissions,
            "permission_overrides": config.permission_overrides,
            "viewport_width": config.viewport_width,
            "viewport_height": config.viewport_height,
            "custom_init_script": bool(config.custom_init_script),
            "headless": config.headless,
            "antidetect_preset": antidetect_preset,
        }

        return {
            "success": True,
            "current_page_id": state.current_page_id,
            "applied": applied,
        }


@mcp.tool()
async def apply_antidetect_preset(ctx: Context, preset: str = "pixelscan") -> Dict[str, Any]:
    """Apply an anti-detect preset and rebuild the context."""

    return await recreate_context(ctx=ctx, antidetect_preset=preset)


@mcp.tool()
async def get_effective_config(ctx: Context) -> Dict[str, Any]:
    """Return the currently applied fingerprint/stealth/browser settings."""

    state = get_browser_state(ctx)
    return {
        "headless": config.headless,
        "browser_type": config.browser_type,
        "channel": config.channel,
        "stealth": config.stealth,
        "mask_devtools": config.mask_devtools,
        "force_devtools_closed": config.force_devtools_closed,
        "apply_cdp_patch": config.apply_cdp_patch,
        "override_viewport": config.override_viewport,
        "viewport_width": config.viewport_width,
        "viewport_height": config.viewport_height,
        "device_scale_factor": config.device_scale_factor,
        "user_agent": config.user_agent,
        "accept_language": config.accept_language,
        "languages": config.languages,
        "locale": config.locale,
        "timezone_id": config.timezone_id,
        "platform": config.platform,
        "vendor": config.vendor,
        "hardware_concurrency": config.hardware_concurrency,
        "device_memory": config.device_memory,
        "sec_ch_ua": config.sec_ch_ua,
        "sec_ch_ua_mobile": config.sec_ch_ua_mobile,
        "sec_ch_ua_platform": config.sec_ch_ua_platform,
        "sec_ch_ua_platform_version": config.sec_ch_ua_platform_version,
        "sec_ch_ua_full_version_list": config.sec_ch_ua_full_version_list,
        "fill_plugins": config.fill_plugins,
        "override_languages": config.override_languages,
        "stub_chrome_runtime": config.stub_chrome_runtime,
        "permission_overrides": config.permission_overrides,
        "grant_permissions": config.grant_permissions,
        "permissions_origin": config.permissions_origin,
        "current_page_id": state.current_page_id if state else None,
    }


@mcp.tool()
async def export_playwright_context(ctx: Context) -> Dict[str, Any]:
    """Export the current Playwright launch/context options and init script for reuse outside MCP."""

    state = get_browser_state(ctx)
    headers = _build_extra_headers(config)
    locale_value = config.locale or (config.languages[0] if config.languages else None)

    launch_options: Dict[str, Any] = {
        "headless": config.headless,
    }
    if config.channel:
        launch_options["channel"] = config.channel

    context_options: Dict[str, Any] = {}
    if config.override_viewport:
        context_options["viewport"] = {
            "width": config.viewport_width,
            "height": config.viewport_height,
        }
        if config.device_scale_factor is not None:
            context_options["device_scale_factor"] = config.device_scale_factor
            context_options.setdefault(
                "screen",
                {"width": config.viewport_width, "height": config.viewport_height},
            )
    if config.user_agent:
        context_options["user_agent"] = config.user_agent
    if locale_value:
        context_options["locale"] = locale_value
    if config.timezone_id:
        context_options["timezone_id"] = config.timezone_id
    if headers:
        context_options["extra_http_headers"] = headers
    if config.grant_permissions:
        context_options["permissions"] = config.grant_permissions

    init_script = _build_context_init_script(config) or ""

    js_snippet = f"""// Launch
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({json.dumps(launch_options, indent=2)});
  const context = await browser.newContext({json.dumps(context_options, indent=2)});
  {'await context.addInitScript(`' + init_script.replace('`', '\\`') + '`);' if init_script else '// No init script applied'}
  const page = await context.newPage();
  // ... your automation here ...
  await browser.close();
}})();
"""

    py_snippet = f"""# Launch
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch({json.dumps(launch_options, indent=2)})
    context = browser.new_context({json.dumps(context_options, indent=2)})
    {"context.add_init_script('''" + init_script.replace("'''", "''") + "''')" if init_script else "# No init script applied"}
    page = context.new_page()
    # ... your automation here ...
    browser.close()
"""

    return {
        "launch_options": launch_options,
        "context_options": context_options,
        "init_script": init_script,
        "flags": {
            "stealth": config.stealth,
            "mask_devtools": config.mask_devtools,
            "force_devtools_closed": config.force_devtools_closed,
            "apply_cdp_patch": config.apply_cdp_patch,
            "fill_plugins": config.fill_plugins,
            "override_languages": config.override_languages,
            "stub_chrome_runtime": config.stub_chrome_runtime,
            "override_viewport": config.override_viewport,
        },
        "example_js": js_snippet,
        "example_python": py_snippet,
        "current_page_id": state.current_page_id if state else None,
    }


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
        page = get_current_page(ctx)
        await page.goto(url)
        current_url = page.url
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
        page = get_current_page(ctx)
        await page.reload()
        current_url = page.url
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
        page = get_current_page(ctx)
        await page.go_back()
        current_url = page.url
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
        page = get_current_page(ctx)
        await page.go_forward()
        current_url = page.url
        return NavigationResult(success=True, url=current_url)
    except Exception as e:
        return NavigationResult(success=False, url="", error=str(e))


@mcp.tool()
async def get_current_url(ctx: Context) -> CurrentUrlResult:
    """Get the current page URL with parsed components and query parameters.

    This tool retrieves the current page URL and provides parsed information
    including the scheme, domain, path, and query parameters for easy access.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        CurrentUrlResult with success status, URL, parsed components, query parameters, and any errors
    """
    try:
        page = get_current_page(ctx)
        current_url = page.url
        
        # Parse the URL into components
        parsed = urlparse(current_url)
        parsed_url = {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "path": parsed.path,
            "fragment": parsed.fragment,
            "query": parsed.query
        }
        
        # Parse query parameters
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        
        return CurrentUrlResult(
            success=True,
            url=current_url,
            parsed_url=parsed_url,
            query_params=query_params
        )
    except Exception as e:
        return CurrentUrlResult(success=False, error=str(e))


# Page Management Tools
@mcp.tool()
async def list_pages(ctx: Context) -> PageListResult:
    """List all open browser pages/tabs with their information.

    This tool returns information about all currently open pages including their IDs,
    URLs, titles, and which one is currently active. Use this to see all tabs/windows
    that have been opened by user actions or JavaScript.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        PageListResult with list of all pages and their details
    """
    try:
        browser_state = get_browser_state(ctx)
        pages_info = []

        for page_id, page in browser_state.pages.items():
            try:
                pages_info.append(PageInfo(
                    page_id=page_id,
                    url=page.url,
                    title=await page.title(),
                    is_current=(page_id == browser_state.current_page_id)
                ))
            except Exception as e:
                logger.error(f"Error getting info for page {page_id}: {e}")

        return PageListResult(
            success=True,
            pages=pages_info,
            current_page_id=browser_state.current_page_id
        )
    except Exception as e:
        return PageListResult(success=False, error=str(e))


@mcp.tool()
async def switch_page(page_id: str, ctx: Context) -> PageSwitchResult:
    """Switch to a different browser page/tab by its ID.

    This tool changes the active page that subsequent commands will operate on.
    Use list_pages first to get the available page IDs.

    Args:
        page_id: The ID of the page to switch to
        ctx: MCP context containing the browser state

    Returns:
        PageSwitchResult with success status and the new active page information
    """
    try:
        browser_state = get_browser_state(ctx)

        if page_id not in browser_state.pages:
            return PageSwitchResult(
                success=False,
                error=f"Page with ID {page_id} not found"
            )

        browser_state.set_current_page(page_id)
        page = browser_state.pages[page_id]

        return PageSwitchResult(
            success=True,
            page_id=page_id,
            url=page.url
        )
    except Exception as e:
        return PageSwitchResult(success=False, error=str(e))


@mcp.tool()
async def close_page(page_id: str, ctx: Context) -> Dict[str, Any]:
    """Close a specific browser page/tab by its ID.

    This tool closes the specified page. If it's the current page,
    the tool will automatically switch to another available page.
    Cannot close the last remaining page.

    Args:
        page_id: The ID of the page to close
        ctx: MCP context containing the browser state

    Returns:
        Dict with success status, closed page ID, and any error messages
    """
    try:
        browser_state = get_browser_state(ctx)

        if page_id not in browser_state.pages:
            return {
                "success": False,
                "error": f"Page with ID {page_id} not found"
            }

        if len(browser_state.pages) <= 1:
            return {
                "success": False,
                "error": "Cannot close the last remaining page"
            }

        # Close the page
        page = browser_state.pages[page_id]
        await page.close()

        # Remove from tracking
        del browser_state.pages[page_id]

        # If this was the current page, switch to another
        if browser_state.current_page_id == page_id:
            # Get first available page
            new_page_id = next(iter(browser_state.pages.keys()))
            browser_state.set_current_page(new_page_id)

        return {
            "success": True,
            "closed_page_id": page_id,
            "current_page_id": browser_state.current_page_id
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def wait_for_popup(ctx: Context, timeout: int = 5000) -> NewPageResult:
    """Wait for a new popup window or tab to be opened.

    This tool waits for a new page to be created (e.g., from clicking a link with
    target="_blank" or JavaScript window.open()). It's useful when you know an
    action will open a new window/tab and you want to capture it.

    Args:
        timeout: Maximum time to wait in milliseconds (default: 5000)
        ctx: MCP context containing the browser state

    Returns:
        NewPageResult with information about the new page
    """
    import asyncio
    import uuid

    try:
        browser_state = get_browser_state(ctx)

        # Store the current page ID as the opener
        opener_page_id = browser_state.current_page_id

        # Create a future to wait for the new page
        new_page_future = asyncio.Future()

        def handle_page(page: Page):
            """Handle the new page event."""
            if not new_page_future.done():
                new_page_future.set_result(page)

        # Temporarily listen for new pages
        browser_state.context.once("page", handle_page)

        try:
            # Wait for new page with timeout
            new_page = await asyncio.wait_for(
                new_page_future,
                timeout=timeout / 1000  # Convert to seconds
            )

            # Generate ID for the new page
            page_id = str(uuid.uuid4())

            # Set default timeout for new page
            new_page.set_default_timeout(config.timeout)

            # Add to pages dictionary
            browser_state.pages[page_id] = new_page

            # Set up network monitoring for the new page
            await _setup_network_monitoring_for_page(browser_state, new_page)

            # Wait a bit for the page to load initial content
            try:
                await new_page.wait_for_load_state("domcontentloaded", timeout=1000)
            except:
                pass  # Page might still be loading, that's OK

            return NewPageResult(
                success=True,
                page_id=page_id,
                url=new_page.url,
                opener_page_id=opener_page_id
            )

        except asyncio.TimeoutError:
            return NewPageResult(
                success=False,
                error=f"No new page opened within {timeout}ms"
            )

    except Exception as e:
        return NewPageResult(success=False, error=str(e))


@mcp.tool()
async def switch_to_latest_page(ctx: Context) -> PageSwitchResult:
    """Switch to the most recently opened page/tab.

    This is a convenience tool that switches to the newest page without
    needing to know its ID. Useful after clicking a link that opens
    in a new tab/window.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        PageSwitchResult with success status and the new active page information
    """
    try:
        browser_state = get_browser_state(ctx)

        if not browser_state.pages:
            return PageSwitchResult(
                success=False,
                error="No pages available"
            )

        # Get the last page ID (most recently added)
        latest_page_id = list(browser_state.pages.keys())[-1]

        browser_state.set_current_page(latest_page_id)
        page = browser_state.pages[latest_page_id]

        return PageSwitchResult(
            success=True,
            page_id=latest_page_id,
            url=page.url
        )
    except Exception as e:
        return PageSwitchResult(success=False, error=str(e))


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
        page = get_current_page(ctx)
        await page.click(selector)
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
        page = get_current_page(ctx)
        await page.type(selector, text)
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
        page = get_current_page(ctx)
        await page.fill(selector, value)
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
        page = get_current_page(ctx)
        await page.select_option(selector, value)
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
        page = get_current_page(ctx)
        await page.hover(selector)
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
        page = get_current_page(ctx)
        if selector:
            element = await page.query_selector(selector)
            if element:
                await element.scroll_into_view_if_needed()
                return {"success": True, "selector": selector, "x": x, "y": y}
            else:
                return {
                    "success": False,
                    "selector": selector,
                    "error": "Element not found",
                }
        else:
            await page.mouse.wheel(x, y)
            return {"success": True, "selector": selector, "x": x, "y": y}
    except Exception as e:
        return {"success": False, "selector": selector, "x": x, "y": y, "error": str(e)}


# Element Discovery Tools
@mcp.tool()
async def query_selector(
    selector: str, ctx: Context, max_text_length: Optional[int] = None
) -> ElementQueryResult:
    """Find and return information about the first element matching a selector.

    This tool locates a single element on the page and returns detailed information
    including its tag name, text content, and all attributes.

    Args:
        selector: Playwright selector to find the element (e.g., "#main-header", "button:has-text('Submit')", "[data-testid=login]")
        ctx: MCP context containing the browser state
        max_text_length: Optional override for maximum characters returned for text and attributes (<=0 disables)

    Returns:
        ElementQueryResult with found status, element details (tag, text, attributes), and any errors
    """
    try:
        page = get_current_page(ctx)
        element = await page.query_selector(selector)
        if element:
            # Get element attributes
            tag_name = await element.evaluate("el => el.tagName")
            text_content = await element.evaluate("el => el.textContent")
            attributes = await element.evaluate(
                "el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))"
            )

            text_limit = _resolve_limit(
                config.max_element_text_length, max_text_length
            )
            text_content, text_truncated = _truncate_text(text_content, text_limit)
            attributes, attributes_truncated = _truncate_attributes(
                attributes, text_limit
            )

            element_info: Dict[str, Any] = {
                "tag_name": tag_name,
                "text_content": text_content,
                "attributes": attributes,
            }

            if text_truncated:
                element_info["text_truncated"] = True
            if attributes_truncated:
                element_info["attributes_truncated"] = True

            return ElementQueryResult(
                found=True,
                count=1,
                returned_count=1,
                truncated=text_truncated or attributes_truncated,
                elements=[element_info],
            )
        else:
            return ElementQueryResult(found=False, count=0)
    except Exception as e:
        return ElementQueryResult(found=False, count=0, error=str(e))


@mcp.tool()
async def query_selector_all(
    selector: str,
    ctx: Context,
    max_elements: Optional[int] = None,
    max_text_length: Optional[int] = None,
) -> ElementQueryResult:
    """Find and return information about all elements matching a selector.

    This tool locates all elements on the page that match the selector and returns
    detailed information for each one including tag names, text content, and attributes.

    Args:
        selector: Playwright selector to find elements (e.g., ".nav-item", "input[type=checkbox]", "li")
        ctx: MCP context containing the browser state
        max_elements: Optional override for number of elements returned (<=0 disables)
        max_text_length: Optional override for maximum characters returned per element (<=0 disables)

    Returns:
        ElementQueryResult with found status, count, array of element details, and any errors
    """
    try:
        page = get_current_page(ctx)
        elements = await page.query_selector_all(selector)

        total_count = len(elements)
        element_limit = _resolve_limit(config.max_elements_returned, max_elements)
        if element_limit > 0:
            elements = elements[:element_limit]

        elements_info = []
        any_truncated = total_count > len(elements)

        for element in elements:
            tag_name = await element.evaluate("el => el.tagName")
            text_content = await element.evaluate("el => el.textContent")
            attributes = await element.evaluate(
                "el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))"
            )

            text_limit = _resolve_limit(
                config.max_element_text_length, max_text_length
            )
            text_content, text_truncated = _truncate_text(text_content, text_limit)
            attributes, attributes_truncated = _truncate_attributes(
                attributes, text_limit
            )

            if text_truncated or attributes_truncated:
                any_truncated = True

            element_info: Dict[str, Any] = {
                "tag_name": tag_name,
                "text_content": text_content,
                "attributes": attributes,
            }

            if text_truncated:
                element_info["text_truncated"] = True
            if attributes_truncated:
                element_info["attributes_truncated"] = True

            elements_info.append(element_info)

        return ElementQueryResult(
            found=total_count > 0,
            count=total_count,
            returned_count=len(elements_info),
            elements=elements_info,
            truncated=any_truncated,
        )
    except Exception as e:
        return ElementQueryResult(found=False, count=0, error=str(e))


@mcp.tool()
async def query_selector_meta(
    selector: str,
    ctx: Context,
    preview_length: int = 200,
    max_elements: Optional[int] = None,
) -> ElementMetaResult:
    """Return lightweight metadata for elements matching a selector.

    Provides tag name, role, a short text preview, and key attributes without
    dumping full text content. Useful for scouting before fetching full details.

    Args:
        selector: Playwright selector to find elements
        ctx: MCP context containing the browser state
        preview_length: Maximum characters to include in the text preview (<=0 disables)
        max_elements: Optional override for number of elements returned (<=0 disables)
    """

    try:
        page = get_current_page(ctx)
        elements = await page.query_selector_all(selector)

        total_count = len(elements)
        element_limit = _resolve_limit(config.max_elements_returned, max_elements)
        if element_limit > 0:
            elements = elements[:element_limit]

        elements_info = []
        any_truncated = total_count > len(elements)

        preview_limit = preview_length if preview_length is not None else 200

        for element in elements:
            element_data = await element.evaluate(
                "el => ({\n                    tagName: el.tagName,\n                    role: el.getAttribute('role'),\n                    textContent: el.textContent,\n                    attributes: Array.from(el.attributes).reduce((acc, attr) => { acc[attr.name] = attr.value; return acc; }, {})\n                })"
            )

            raw_text = element_data.get("textContent") or ""
            if preview_limit <= 0:
                text_preview = ""
                text_truncated = len(raw_text) > 0
            else:
                text_preview, text_truncated = _truncate_text(
                    raw_text, preview_limit
                )

            raw_attributes = element_data.get("attributes") or {}
            filtered_attributes = {
                key: value
                for key, value in raw_attributes.items()
                if key in META_ATTRIBUTE_KEYS and value is not None
            }

            if element_data.get("role") and "role" not in filtered_attributes:
                filtered_attributes["role"] = element_data["role"]

            element_info = {
                "tag_name": element_data.get("tagName"),
                "role": element_data.get("role"),
                "text_preview": text_preview,
                "attributes": filtered_attributes,
                "preview_length": preview_limit,
            }

            if text_truncated:
                element_info["text_truncated"] = True
                any_truncated = True

            elements_info.append(element_info)

        return ElementMetaResult(
            found=total_count > 0,
            count=total_count,
            returned_count=len(elements_info),
            elements=elements_info,
            truncated=any_truncated,
        )
    except Exception as e:
        return ElementMetaResult(found=False, count=0, error=str(e))


# Snapshotting Tools
@mcp.tool()
async def get_html(ctx: Context) -> Dict[str, Any]:
    """Retrieve the complete HTML source of the current page.

    This tool returns the page HTML (doctype, head, body). The payload is trimmed
    to the configured response budget and linked to an overflow artifact when the
    full document would exceed inline limits.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        Dict with success status and the complete HTML source as a string
    """
    try:
        page = get_current_page(ctx)
        html = await page.content()
        (
            budgeted_html,
            truncated,
            preview,
            overflow_path,
            original_size,
        ) = _apply_response_budget(
            html,
            budget=config.max_response_characters,
            preview_limit=config.preview_characters,
            label="page_html",
        )

        result: Dict[str, Any] = {
            "success": True,
            "html": budgeted_html,
            "original_length": len(html),
        }

        if truncated:
            result["truncated"] = True
            result["preview"] = preview
            result["overflow_characters"] = original_size
            if overflow_path:
                result["overflow_path"] = overflow_path

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_accessibility_snapshot(
    ctx: Context,
    interesting_only: bool = True,
    root_selector: Optional[str] = None,
    max_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Capture the accessibility tree structure of the current page.

    This tool returns the accessibility tree used by screen readers and other
    assistive technologies, including roles, names, and hierarchical structure.

    Args:
        ctx: MCP context containing the browser state
        interesting_only: Whether to return only nodes Playwright deems interesting (default True)
        root_selector: Optional selector to scope the snapshot to a specific subtree
        max_nodes: Optional override for the node cap applied to the returned snapshot (<=0 disables)

    Returns:
        Dict with success status and the accessibility tree snapshot structure
    """
    try:
        page = get_current_page(ctx)

        root_element = None
        if root_selector:
            root_element = await page.query_selector(root_selector)
            if root_element is None:
                return {
                    "success": False,
                    "error": "Root selector not found",
                    "selector": root_selector,
                }

        snapshot = await page.accessibility.snapshot(
            interesting_only=interesting_only,
            root=root_element,
        )

        node_limit = _resolve_limit(config.max_accessibility_nodes, max_nodes)
        pruned_snapshot, node_truncated, node_count = _prune_accessibility_snapshot(
            snapshot, node_limit
        )

        (
            budgeted_snapshot,
            budget_truncated,
            preview,
            overflow_path,
            original_size,
        ) = _apply_response_budget(
            pruned_snapshot,
            budget=config.max_response_characters,
            preview_limit=config.preview_characters,
            label="accessibility_snapshot",
        )

        result: Dict[str, Any] = {
            "success": True,
            "snapshot": budgeted_snapshot,
            "node_count": node_count,
            "interesting_only": interesting_only,
        }

        if root_selector:
            result["root_selector"] = root_selector

        if node_truncated:
            result["truncated"] = True
            result["max_nodes"] = node_limit

        if budget_truncated:
            result["truncated"] = True
            result["preview"] = preview
            if overflow_path:
                result["overflow_path"] = overflow_path
            result["overflow_characters"] = original_size

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def screenshot(
    ctx: Context,
    selector: Optional[str] = None,
    full_page: bool = False,
    inline: bool = False,
    preview_bytes: int = 1024,
) -> ScreenshotResult:
    """Capture a screenshot of the page or a specific element.

    This tool captures either the viewport, the full page, or a specific element.
    By default it stores the PNG in the artifact directory and returns metadata plus
    a short inline preview. Set ``inline=True`` to embed the full base64 payload like
    the legacy behaviour.

    Args:
        ctx: MCP context containing the browser state
        selector: Optional Playwright selector for a specific element to screenshot
        full_page: If True, captures the entire page including content below the fold
        inline: When True, return the entire screenshot as base64 (may consume many tokens)
        preview_bytes: Number of bytes to include in the inline preview when not in inline mode

    Returns:
        ScreenshotResult with success status plus either inline base64 data or an
        ``artifact_path`` pointing at the saved PNG.
    """
    try:
        page = get_current_page(ctx)

        dimensions: Optional[Dict[str, Union[int, float]]] = None

        if selector:
            element = await page.query_selector(selector)
            if not element:
                return ScreenshotResult(success=False, error="Element not found")
            screenshot_bytes = await element.screenshot()
            try:
                box = await element.bounding_box()
            except Exception:  # pragma: no cover - best effort only
                box = None
            if box:
                dimensions = {"width": box.get("width"), "height": box.get("height")}
        else:
            screenshot_bytes = await page.screenshot(full_page=full_page)
            viewport = page.viewport_size or {}
            if viewport:
                dimensions = {
                    "width": viewport.get("width"),
                    "height": viewport.get("height"),
                }

        byte_size = len(screenshot_bytes)
        sha256 = hashlib.sha256(screenshot_bytes).hexdigest()

        if inline:
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            preview = (
                screenshot_base64[: config.preview_characters]
                if config.preview_characters > 0
                else None
            )
            return ScreenshotResult(
                success=True,
                data=screenshot_base64,
                format="png",
                byte_size=byte_size,
                sha256=sha256,
                preview_base64=preview,
                dimensions=dimensions,
                inline=True,
            )

        preview_slice = screenshot_bytes[: max(0, min(preview_bytes, byte_size))]
        preview_base64 = (
            base64.b64encode(preview_slice).decode("utf-8")
            if preview_slice
            else None
        )

        artifact_path = _make_artifact_path("screenshot", suffix=".png")
        try:
            artifact_path.write_bytes(screenshot_bytes)
            _enforce_artifact_retention()
        except Exception as exc:
            return ScreenshotResult(success=False, error=str(exc))

        return ScreenshotResult(
            success=True,
            format="png",
            artifact_path=str(artifact_path),
            byte_size=byte_size,
            sha256=sha256,
            preview_base64=preview_base64,
            dimensions=dimensions,
            inline=False,
        )
    except Exception as e:
        return ScreenshotResult(success=False, error=str(e))


@mcp.tool()
async def pdf(
    ctx: Context, inline: bool = False, preview_bytes: int = 2048
) -> PDFResult:
    """Generate a PDF document from the current page.

    This tool converts the current page to a PDF (Chromium only). By default the
    document is saved to the artifact directory and only metadata plus a short
    preview are returned. Set ``inline=True`` to embed the entire base64 payload
    when absolutely necessary.

    Args:
        ctx: MCP context containing the browser state
        inline: When True, return the full base64 PDF inline (large responses)
        preview_bytes: Number of bytes to expose in the inline preview when not embedding

    Returns:
        PDFResult with success status and either inline data or an ``artifact_path``
        referencing the saved PDF.
    """
    try:
        page = get_current_page(ctx)
        pdf_bytes = await page.pdf()
        byte_size = len(pdf_bytes)
        sha256 = hashlib.sha256(pdf_bytes).hexdigest()

        if inline:
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
            preview = (
                pdf_base64[: config.preview_characters]
                if config.preview_characters > 0
                else None
            )
            return PDFResult(
                success=True,
                data=pdf_base64,
                byte_size=byte_size,
                sha256=sha256,
                preview_base64=preview,
                inline=True,
            )

        preview_slice = pdf_bytes[: max(0, min(preview_bytes, byte_size))]
        preview_base64 = (
            base64.b64encode(preview_slice).decode("utf-8")
            if preview_slice
            else None
        )

        artifact_path = _make_artifact_path("pdf", suffix=".pdf")
        try:
            artifact_path.write_bytes(pdf_bytes)
            _enforce_artifact_retention()
        except Exception as exc:
            return PDFResult(success=False, error=str(exc))

        return PDFResult(
            success=True,
            artifact_path=str(artifact_path),
            byte_size=byte_size,
            sha256=sha256,
            preview_base64=preview_base64,
            inline=False,
        )
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
        page = get_current_page(ctx)
        result = await page.evaluate(script)
        (
            processed_result,
            truncated,
            preview,
            overflow_path,
            original_size,
        ) = _apply_response_budget(
            result,
            budget=config.max_response_characters,
            preview_limit=config.preview_characters,
            label="evaluate_result",
        )

        payload: Dict[str, Any] = {
            "success": True,
            "result": processed_result,
        }

        if truncated:
            payload["truncated"] = True
            payload["preview"] = preview
            if overflow_path:
                payload["overflow_path"] = overflow_path
            payload["overflow_characters"] = original_size

        return ScriptResult(**payload)
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
        page = get_current_page(ctx)
        element = await page.query_selector(selector)
        if element:
            visible = await element.is_visible()
            return {"success": True, "selector": selector, "visible": visible}
        else:
            return {
                "success": False,
                "selector": selector,
                "error": "Element not found",
            }
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
        page = get_current_page(ctx)
        element = await page.query_selector(selector)
        if element:
            enabled = await element.is_enabled()
            return {"success": True, "selector": selector, "enabled": enabled}
        else:
            return {
                "success": False,
                "selector": selector,
                "error": "Element not found",
            }
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def wait_for_element(
    selector: str, ctx: Context, timeout: int = 30000
) -> Dict[str, Any]:
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
        page = get_current_page(ctx)
        await page.wait_for_selector(selector, timeout=timeout)
        return {"success": True, "selector": selector, "timeout": timeout}
    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "timeout": timeout,
            "error": str(e),
        }


@mcp.tool()
async def wait_for_load_state(
    state: str, ctx: Context, timeout: int = 30000
) -> Dict[str, Any]:
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
        page = get_current_page(ctx)
        await page.wait_for_load_state(state, timeout=timeout)
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
        page = get_current_page(ctx)
        await page.fill(selector, "")
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
        page = get_current_page(ctx)
        await page.check(selector)
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
        page = get_current_page(ctx)
        await page.uncheck(selector)
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
        page = get_current_page(ctx)
        await page.set_input_files(selector, file_path)
        return {"success": True, "selector": selector, "file_path": file_path}
    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "file_path": file_path,
            "error": str(e),
        }


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
        page = get_current_page(ctx)
        await page.keyboard.press(key)
        return {"success": True, "key": key}
    except Exception as e:
        return {"success": False, "key": key, "error": str(e)}


# Advanced Navigation Tools
@mcp.tool()
async def wait_for_url(
    url_pattern: str, ctx: Context, timeout: int = 30000
) -> Dict[str, Any]:
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
        page = get_current_page(ctx)
        await page.wait_for_url(url_pattern, timeout=timeout)
        current_url = page.url
        return {
            "success": True,
            "url_pattern": url_pattern,
            "current_url": current_url,
            "timeout": timeout,
        }
    except Exception as e:
        return {
            "success": False,
            "url_pattern": url_pattern,
            "timeout": timeout,
            "error": str(e),
        }


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
        page = get_current_page(ctx)
        await page.set_viewport_size({"width": width, "height": height})
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
        page = get_current_page(ctx)
        element = await page.query_selector(selector)
        if element:
            box = await element.bounding_box()
            return {"success": True, "selector": selector, "bounding_box": box}
        else:
            return {
                "success": False,
                "selector": selector,
                "error": "Element not found",
            }
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
        page = get_current_page(ctx)
        element = await page.query_selector(selector)
        if element:
            attributes = await element.evaluate(
                "el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))"
            )
            return {"success": True, "selector": selector, "attributes": attributes}
        else:
            return {
                "success": False,
                "selector": selector,
                "error": "Element not found",
            }
    except Exception as e:
        return {"success": False, "selector": selector, "error": str(e)}


@mcp.tool()
async def get_computed_style(
    selector: str, property: str, ctx: Context
) -> Dict[str, Any]:
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
        page = get_current_page(ctx)
        element = await page.query_selector(selector)
        if element:
            style_value = await element.evaluate(
                f"el => getComputedStyle(el).{property}"
            )
            return {
                "success": True,
                "selector": selector,
                "property": property,
                "value": style_value,
            }
        else:
            return {
                "success": False,
                "selector": selector,
                "error": "Element not found",
            }
    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "property": property,
            "error": str(e),
        }


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
        page = get_current_page(ctx)
        await page.wait_for_load_state("networkidle", timeout=timeout)
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
        page = get_current_page(ctx)
        # Note: This would require setting up error listeners during browser initialization
        # For now, we'll evaluate to check for any stored errors
        errors = await page.evaluate(
            """
            () => {
                if (window.pageErrors) {
                    return window.pageErrors;
                }
                return [];
            }
        """
        )
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
        page = get_current_page(ctx)
        # Note: This would require setting up console listeners during browser initialization
        # For now, we'll return a message about setup requirements
        logs = await page.evaluate(
            """
            () => {
                if (window.consoleLogs) {
                    return window.consoleLogs;
                }
                return [];
            }
        """
        )
        return {
            "success": True,
            "logs": logs,
            "note": "Console logging requires browser setup with listeners",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Network Monitoring & Interception Tools
@mcp.tool()
async def get_network_requests(
    ctx: Context, url_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve captured network requests with optional URL filtering.

    This tool returns all network requests that have been captured since the page
    started loading, including details like method, headers, and post data.

    Args:
        ctx: MCP context containing the browser state
        url_pattern: Optional glob pattern to filter requests by URL (e.g., "*/api/*", "https://example.com/*")

    Returns:
        Dict with success status, array of request details, and any errors
    """
    try:
        import fnmatch

        browser_state = get_browser_state(ctx)
        requests = browser_state.captured_requests

        if url_pattern:
            requests = [
                req for req in requests if fnmatch.fnmatch(req["url"], url_pattern)
            ]

        total_count = len(requests)
        (
            clipped_requests,
            truncated,
            preview,
            overflow_path,
            original_size,
        ) = _apply_response_budget(
            requests,
            budget=config.max_response_characters,
            preview_limit=config.preview_characters,
            label="network_requests",
        )

        result: Dict[str, Any] = {
            "success": True,
            "requests": clipped_requests,
            "count": total_count,
        }

        if isinstance(clipped_requests, list):
            result["returned_count"] = len(clipped_requests)

        if truncated:
            result["truncated"] = True
            result["preview"] = preview
            if overflow_path:
                result["overflow_path"] = overflow_path
            result["overflow_characters"] = original_size

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_network_responses(
    ctx: Context, url_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve captured network responses with optional URL filtering.

    This tool returns all network responses that have been captured since the page
    started loading, including status codes, headers, and response details.

    Args:
        ctx: MCP context containing the browser state
        url_pattern: Optional glob pattern to filter responses by URL (e.g., "*/api/*", "*.json")

    Returns:
        Dict with success status, array of response details, and any errors
    """
    try:
        import fnmatch

        browser_state = get_browser_state(ctx)
        responses = browser_state.captured_responses

        if url_pattern:
            responses = [
                resp for resp in responses if fnmatch.fnmatch(resp["url"], url_pattern)
            ]

        total_count = len(responses)
        (
            clipped_responses,
            truncated,
            preview,
            overflow_path,
            original_size,
        ) = _apply_response_budget(
            responses,
            budget=config.max_response_characters,
            preview_limit=config.preview_characters,
            label="network_responses",
        )

        result: Dict[str, Any] = {
            "success": True,
            "responses": clipped_responses,
            "count": total_count,
        }

        if isinstance(clipped_responses, list):
            result["returned_count"] = len(clipped_responses)

        if truncated:
            result["truncated"] = True
            result["preview"] = preview
            if overflow_path:
                result["overflow_path"] = overflow_path
            result["overflow_characters"] = original_size

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def clear_network_logs(ctx: Context) -> Dict[str, Any]:
    """Clear all captured network request and response logs.

    This tool resets the network monitoring logs, useful for starting fresh
    network monitoring for specific page interactions.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        Dict with success status and cleared counts
    """
    try:
        browser_state = get_browser_state(ctx)
        request_count = len(browser_state.captured_requests)
        response_count = len(browser_state.captured_responses)

        browser_state.captured_requests.clear()
        browser_state.captured_responses.clear()
        browser_state.response_handles.clear()
        browser_state.response_handle_order.clear()

        return {
            "success": True,
            "cleared_requests": request_count,
            "cleared_responses": response_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def intercept_route(
    ctx: Context,
    url_pattern: str,
    action: str,
    status_code: Optional[int] = 200,
    response_body: Optional[str] = None,
    response_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Intercept network requests matching a URL pattern and handle them.

    This tool allows intercepting network requests and either blocking them,
    fulfilling them with custom responses, or modifying them before they continue.

    Args:
        ctx: MCP context containing the browser state
        url_pattern: Glob pattern to match URLs (e.g., "**/api/users", "https://example.com/*")
        action: Action to take - "block", "fulfill", or "continue"
        status_code: HTTP status code for fulfilled responses (default: 200)
        response_body: Response body for fulfilled requests
        response_headers: Custom headers for fulfilled responses

    Returns:
        Dict with success status, intercepted pattern, action, and any errors
    """
    try:
        page = get_current_page(ctx)

        async def route_handler(route: Route, request: Request):
            if action == "block":
                await route.abort()
            elif action == "fulfill":
                await route.fulfill(
                    status=status_code or 200,
                    body=response_body or "",
                    headers=response_headers or {},
                )
            else:  # continue
                await route.continue_()

        await page.route(url_pattern, route_handler)

        return {
            "success": True,
            "url_pattern": url_pattern,
            "action": action,
            "status_code": status_code,
        }
    except Exception as e:
        return {"success": False, "url_pattern": url_pattern, "error": str(e)}


@mcp.tool()
async def unroute_all(ctx: Context) -> Dict[str, Any]:
    """Remove all route interceptors.

    This tool clears all previously set up route interceptors, allowing
    network requests to proceed normally.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        Dict with success status and any errors
    """
    try:
        page = get_current_page(ctx)
        await page.unroute_all()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def wait_for_response(
    ctx: Context, url_pattern: str, timeout: int = 30000
) -> Dict[str, Any]:
    """Wait for a network response matching the specified URL pattern.

    This tool waits for a specific network response to occur, useful for
    waiting for API calls or resource loading to complete.

    Args:
        ctx: MCP context containing the browser state
        url_pattern: Glob pattern or URL to wait for (e.g., "**/api/data", "https://example.com/endpoint")
        timeout: Maximum wait time in milliseconds (default: 30000)

    Returns:
        Dict with success status, response details (URL, status), and any errors
    """
    try:
        page = get_current_page(ctx)
        response = await page.wait_for_response(
            url_pattern, timeout=timeout
        )

        return {
            "success": True,
            "url": response.url,
            "status": response.status,
            "status_text": response.status_text,
            "headers": await response.all_headers(),
        }
    except Exception as e:
        return {
            "success": False,
            "url_pattern": url_pattern,
            "timeout": timeout,
            "error": str(e),
        }


@mcp.tool()
async def get_response_body(ctx: Context, url_pattern: str) -> Dict[str, Any]:
    """Get the response body for the most recent response matching a URL pattern.

    This tool retrieves network response bodies. Text payloads are trimmed with
    the shared response budget; binary bodies are written to the artifact store
    and returned by reference.

    Args:
        ctx: MCP context containing the browser state
        url_pattern: Glob pattern to match response URLs

    Returns:
        Dict with success status, response body (as text), and any errors
    """
    try:
        import fnmatch

        browser_state = get_browser_state(ctx)

        matching_responses = [
            resp
            for resp in reversed(browser_state.captured_responses)
            if fnmatch.fnmatch(resp["url"], url_pattern)
        ]

        if not matching_responses:
            return {"success": False, "error": "No matching responses found"}

        response_info = matching_responses[0]
        handle_id = response_info.get("handle_id")
        response_obj = browser_state.response_handles.get(handle_id) if handle_id else None

        if response_obj is None:
            return {
                "success": False,
                "error": "No live response handle available. Re-run the request before fetching the body.",
            }

        body_bytes = await response_obj.body()
        headers = await response_obj.all_headers()
        content_type = headers.get("content-type", "")
        if not content_type:
            content_type = response_info.get("headers", {}).get("content-type", "")

        byte_size = len(body_bytes)
        sha256 = hashlib.sha256(body_bytes).hexdigest()

        if _is_text_content(content_type):
            text = body_bytes.decode("utf-8", errors="replace")
            (
                processed_text,
                truncated,
                preview,
                overflow_path,
                original_size,
            ) = _apply_response_budget(
                text,
                budget=config.max_response_characters,
                preview_limit=config.preview_characters,
                label="response_body",
            )

            result: Dict[str, Any] = {
                "success": True,
                "url": response_obj.url,
                "body": processed_text,
                "content_type": content_type,
                "byte_size": byte_size,
                "sha256": sha256,
                "headers": headers,
            }

            if truncated:
                result["truncated"] = True
                result["preview"] = preview
                result["overflow_characters"] = original_size
                if overflow_path:
                    result["overflow_path"] = overflow_path
            return result

        suffix = _guess_suffix_from_mime(content_type, ".bin")
        artifact_path = _make_artifact_path("response_body", suffix=suffix)
        try:
            artifact_path.write_bytes(body_bytes)
            _enforce_artifact_retention()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        preview_slice = body_bytes[: min(byte_size, 512)]
        preview_base64 = (
            base64.b64encode(preview_slice).decode("utf-8") if preview_slice else None
        )

        return {
            "success": True,
            "url": response_obj.url,
            "content_type": content_type,
            "byte_size": byte_size,
            "sha256": sha256,
            "artifact_path": str(artifact_path),
            "preview_base64": preview_base64,
            "binary": True,
            "headers": headers,
        }
    except Exception as e:
        return {"success": False, "url_pattern": url_pattern, "error": str(e)}


def _read_artifact_chunk_impl(path: Path, offset: int, limit: int) -> Dict[str, Any]:
    """Read a bounded chunk from an artifact file."""

    byte_size = max(path.stat().st_size, 0)
    start = min(max(offset, 0), byte_size)
    chunk_size = max(limit, 1)

    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(chunk_size)

    is_binary = _is_probably_binary(data) if data else path.suffix not in {".txt", ".json", ".log"}

    payload: Dict[str, Any] = {
        "path": str(path),
        "offset": start,
        "bytes_read": len(data),
        "byte_size": byte_size,
        "binary": bool(is_binary),
        "next_offset": start + len(data) if start + len(data) < byte_size else None,
    }

    if not data:
        payload["content"] = ""
        payload["binary"] = False
        return payload

    if is_binary:
        payload["content_base64"] = base64.b64encode(data).decode("utf-8")
    else:
        payload["content"] = data.decode("utf-8", errors="replace")

    return payload


@mcp.tool()
async def read_artifact(path: str, ctx: Context) -> Dict[str, Any]:
    """Read an artifact produced by response budgeting.

    The first chunk of the artifact is returned using the configured inline budget.
    Use :func:`read_artifact_chunk` for additional data when ``truncated`` is true.
    """

    del ctx  # Maintains consistent tool signature; context not needed here.

    limit = config.max_response_characters
    if limit <= 0:
        limit = config.artifact_chunk_size

    try:
        resolved_path = _resolve_artifact_path(path)
        payload = _read_artifact_chunk_impl(resolved_path, offset=0, limit=limit)
        truncated = payload.get("next_offset") is not None
        result: Dict[str, Any] = {"success": True, **payload}
        if truncated:
            result["truncated"] = True
            result["message"] = "Content truncated. Use read_artifact_chunk for additional data."
        return result
    except (ValueError, PermissionError, FileNotFoundError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"success": False, "error": str(exc)}


@mcp.tool()
async def read_artifact_chunk(
    path: str, ctx: Context, offset: int = 0, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Read a bounded chunk from an artifact file.

    Returns decoded text when the bytes look textual, otherwise base64 payloads so
    binary content stays compact. ``next_offset`` indicates where to resume.
    """

    del ctx  # Tool does not need the browser context.

    try:
        resolved_path = _resolve_artifact_path(path)
        chunk_limit = limit if limit and limit > 0 else config.artifact_chunk_size
        payload = _read_artifact_chunk_impl(resolved_path, offset=offset, limit=chunk_limit)
        return {"success": True, **payload}
    except (ValueError, PermissionError, FileNotFoundError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"success": False, "error": str(exc)}


# Cookie Management Tools
@mcp.tool()
async def get_cookies(ctx: Context, urls: Optional[List[str]] = None) -> CookieResult:
    """Retrieve cookies from the browser context.

    This tool returns cookies from the current browser context, with optional
    filtering by specific URLs or domains.

    Args:
        ctx: MCP context containing the browser state
        urls: Optional list of URLs to filter cookies by domain

    Returns:
        CookieResult with success status, array of cookie objects, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        cookies = await browser_state.context.cookies(urls or [])
        return CookieResult(success=True, cookies=cookies)
    except Exception as e:
        return CookieResult(success=False, error=str(e))


@mcp.tool()
async def add_cookies(ctx: Context, cookies: List[Dict[str, Any]]) -> CookieResult:
    """Add cookies to the browser context.

    This tool adds one or more cookies to the current browser context,
    making them available for all pages in the context.

    Args:
        ctx: MCP context containing the browser state
        cookies: List of cookie objects with name, value, domain, path, etc.
                Example: [{"name": "session", "value": "abc123", "domain": "example.com"}]

    Returns:
        CookieResult with success status and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        await browser_state.context.add_cookies(cookies)
        return CookieResult(success=True, cookies=cookies)
    except Exception as e:
        return CookieResult(success=False, error=str(e))


@mcp.tool()
async def clear_cookies(
    ctx: Context, name: Optional[str] = None, domain: Optional[str] = None
) -> CookieResult:
    """Clear cookies from the browser context.

    This tool removes cookies from the current browser context, with optional
    filtering by name or domain to clear specific cookies.

    Args:
        ctx: MCP context containing the browser state
        name: Optional cookie name to clear specific cookie
        domain: Optional domain to clear cookies from specific domain

    Returns:
        CookieResult with success status and any errors
    """
    try:
        browser_state = get_browser_state(ctx)

        # Build the filter for clearing cookies
        clear_filter = {}
        if name:
            clear_filter["name"] = name
        if domain:
            clear_filter["domain"] = domain

        await browser_state.context.clear_cookies(**clear_filter)

        return CookieResult(success=True, cookies=[{"cleared_filter": clear_filter}])
    except Exception as e:
        return CookieResult(success=False, error=str(e))


# Storage Management Tools
@mcp.tool()
async def get_local_storage(
    ctx: Context, origin: Optional[str] = None
) -> StorageResult:
    """Retrieve localStorage data from the current page or specified origin.

    This tool accesses browser localStorage data, useful for inspecting
    client-side stored data and application state.

    Args:
        ctx: MCP context containing the browser state
        origin: Optional origin URL to get storage from (defaults to current page)

    Returns:
        StorageResult with success status, localStorage data as dict, and any errors
    """
    try:
        browser_state = get_browser_state(ctx)
        page = get_current_page(ctx)

        if origin:
            # Get storage for specific origin
            storage_state = await browser_state.context.storage_state()
            origin_storage = {}
            for origin_data in storage_state.get("origins", []):
                if origin_data["origin"] == origin:
                    origin_storage = {
                        item["name"]: item["value"]
                        for item in origin_data.get("localStorage", [])
                    }
                    break
            data = origin_storage
        else:
            # Get localStorage for current page
            data = await page.evaluate(
                """
                () => {
                    const storage = {};
                    for (let i = 0; i < localStorage.length; i++) {
                        const key = localStorage.key(i);
                        storage[key] = localStorage.getItem(key);
                    }
                    return storage;
                }
            """
            )

        return StorageResult(success=True, data=data)
    except Exception as e:
        return StorageResult(success=False, error=str(e))


@mcp.tool()
async def set_local_storage(ctx: Context, key: str, value: str) -> StorageResult:
    """Set a localStorage item in the current page.

    This tool stores data in the browser's localStorage for the current page,
    useful for setting up application state or test data.

    Args:
        ctx: MCP context containing the browser state
        key: Storage key name
        value: Storage value to set

    Returns:
        StorageResult with success status, set data, and any errors
    """
    try:
        page = get_current_page(ctx)

        await page.evaluate(
            """
            (args) => localStorage.setItem(args.key, args.value)
        """,
            {"key": key, "value": value},
        )

        return StorageResult(success=True, data={key: value})
    except Exception as e:
        return StorageResult(success=False, error=str(e))


@mcp.tool()
async def get_session_storage(ctx: Context) -> StorageResult:
    """Retrieve sessionStorage data from the current page.

    This tool accesses browser sessionStorage data, useful for inspecting
    session-specific client-side stored data.

    Args:
        ctx: MCP context containing the browser state

    Returns:
        StorageResult with success status, sessionStorage data as dict, and any errors
    """
    try:
        page = get_current_page(ctx)

        data = await page.evaluate(
            """
            () => {
                const storage = {};
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    storage[key] = sessionStorage.getItem(key);
                }
                return storage;
            }
        """
        )

        return StorageResult(success=True, data=data)
    except Exception as e:
        return StorageResult(success=False, error=str(e))


@mcp.tool()
async def set_session_storage(ctx: Context, key: str, value: str) -> StorageResult:
    """Set a sessionStorage item in the current page.

    This tool stores data in the browser's sessionStorage for the current page,
    useful for setting up session-specific test data.

    Args:
        ctx: MCP context containing the browser state
        key: Storage key name
        value: Storage value to set

    Returns:
        StorageResult with success status, set data, and any errors
    """
    try:
        page = get_current_page(ctx)

        await page.evaluate(
            """
            (args) => sessionStorage.setItem(args.key, args.value)
        """,
            {"key": key, "value": value},
        )

        return StorageResult(success=True, data={key: value})
    except Exception as e:
        return StorageResult(success=False, error=str(e))


@mcp.tool()
async def clear_storage(ctx: Context, storage_type: str = "both") -> StorageResult:
    """Clear localStorage and/or sessionStorage from the current page.

    This tool removes stored data from the browser storage, useful for
    resetting application state between test scenarios.

    Args:
        ctx: MCP context containing the browser state
        storage_type: Type of storage to clear - "local", "session", or "both" (default)

    Returns:
        StorageResult with success status, cleared storage types, and any errors
    """
    try:
        # Need the active page to execute storage clearing script
        page = get_current_page(ctx)

        script = ""
        if storage_type in ["local", "both"]:
            script += "localStorage.clear();"
        if storage_type in ["session", "both"]:
            script += "sessionStorage.clear();"

        await page.evaluate(script)

        return StorageResult(success=True, data={"cleared": storage_type})
    except Exception as e:
        return StorageResult(success=False, error=str(e))


# Request Headers & User Agent Tools
@mcp.tool()
async def set_extra_headers(ctx: Context, headers: Dict[str, str]) -> Dict[str, Any]:
    """Set additional HTTP headers for all requests in the browser context.

    This tool adds extra HTTP headers that will be sent with all network requests
    from pages in the current browser context.

    Args:
        ctx: MCP context containing the browser state
        headers: Dictionary of header name-value pairs to add

    Returns:
        Dict with success status, set headers, and any errors
    """
    try:
        page = get_current_page(ctx)
        await page.set_extra_http_headers(headers)

        return {"success": True, "headers": headers}
    except Exception as e:
        return {"success": False, "headers": headers, "error": str(e)}


@mcp.tool()
async def set_user_agent(ctx: Context, user_agent: str) -> Dict[str, Any]:
    """Set the User-Agent header for the browser context.

    This tool changes the User-Agent string that will be sent with requests,
    useful for testing different browser/device behaviors or bypassing restrictions.

    Args:
        ctx: MCP context containing the browser state
        user_agent: User-Agent string to use

    Returns:
        Dict with success status, set user agent, and any errors
    """
    try:
        page = get_current_page(ctx)
        await page.set_user_agent(user_agent)

        return {"success": True, "user_agent": user_agent}
    except Exception as e:
        return {"success": False, "user_agent": user_agent, "error": str(e)}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Playwright MCP Server")
    parser.add_argument("transport", choices=["stdio", "http"], help="Transport type")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport"
    )
    parser.add_argument("--headed", action="store_true", help="Run in headed mode")
    parser.add_argument(
        "--browser",
        choices=["chromium", "firefox", "webkit"],
        default="chromium",
        help="Browser type",
    )
    parser.add_argument(
        "--timeout", type=int, default=30000, help="Default timeout (ms)"
    )
    parser.add_argument(
        "--channel",
        choices=[
            "chrome",
            "chrome-beta",
            "chrome-dev",
            "chrome-canary",
            "msedge",
            "msedge-beta",
            "msedge-dev",
            "msedge-canary",
        ],
        help="Browser channel (use real Chrome/Edge instead of bundled Chromium)",
    )
    parser.add_argument(
        "--user-data-dir",
        type=str,
        help="Path to Chrome user data directory (enables persistent context with your profile)",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=config.max_elements_returned,
        help="Maximum number of elements returned by query selector tools (<=0 disables)",
    )
    parser.add_argument(
        "--max-element-text-length",
        type=int,
        default=config.max_element_text_length,
        help="Maximum characters returned for element text/attributes (<=0 disables)",
    )
    parser.add_argument(
        "--max-accessibility-nodes",
        type=int,
        default=config.max_accessibility_nodes,
        help="Maximum nodes included in accessibility snapshots (<=0 disables)",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=config.max_response_characters,
        help="Maximum characters returned inline before saving overflow to an artifact (<=0 disables)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=config.preview_characters,
        help="Maximum characters included in inline previews for truncated payloads (<=0 disables)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(config.artifact_directory),
        help=(
            "Artifact root directory. A per-session subdirectory is created "
            "automatically to avoid collisions between concurrent Codex/MCP runs"
        ),
    )
    parser.add_argument(
        "--artifact-max-age-seconds",
        type=int,
        default=config.artifact_max_age_seconds,
        help="Maximum age (in seconds) before overflow artifacts are deleted (<=0 disables)",
    )
    parser.add_argument(
        "--artifact-max-files",
        type=int,
        default=config.artifact_max_files,
        help="Maximum number of overflow artifacts to retain (<=0 disables)",
    )
    parser.add_argument(
        "--artifact-chunk-size",
        type=int,
        default=config.artifact_chunk_size,
        help="Default number of bytes returned by artifact chunk reads",
    )
    parser.add_argument(
        "--user-agent",
        dest="user_agent",
        help="User-Agent to present at context creation time",
    )
    parser.add_argument(
        "--accept-language",
        dest="accept_language",
        help="Accept-Language header to send with all requests",
    )
    parser.add_argument(
        "--languages",
        dest="languages",
        help="Comma-separated navigator.languages override (first entry becomes navigator.language when locale is not provided)",
    )
    parser.add_argument(
        "--locale",
        dest="locale",
        help="Context locale (falls back to the first language when omitted)",
    )
    parser.add_argument(
        "--timezone-id",
        dest="timezone_id",
        help="IANA timezone id for the context (e.g., America/Los_Angeles)",
    )
    parser.add_argument(
        "--device-scale-factor",
        dest="device_scale_factor",
        type=float,
        help="Device pixel ratio for the context (affects devicePixelRatio)",
    )
    parser.add_argument(
        "--platform",
        dest="platform",
        help="navigator.platform override (e.g., MacIntel)",
    )
    parser.add_argument(
        "--vendor",
        dest="vendor",
        help="navigator.vendor override (e.g., Google Inc.)",
    )
    parser.add_argument(
        "--hardware-concurrency",
        dest="hardware_concurrency",
        type=int,
        help="navigator.hardwareConcurrency override",
    )
    parser.add_argument(
        "--device-memory",
        dest="device_memory",
        type=float,
        help="navigator.deviceMemory override",
    )
    parser.add_argument(
        "--sec-ch-ua",
        dest="sec_ch_ua",
        help='Sec-CH-UA header value (e.g., "\\"Chromium\\";v=\\"120\\", \\"Google Chrome\\";v=\\"120\\"")',
    )
    parser.add_argument(
        "--sec-ch-ua-full-version-list",
        dest="sec_ch_ua_full_version_list",
        help="Sec-CH-UA-Full-Version-List header value",
    )
    parser.add_argument(
        "--sec-ch-ua-mobile",
        dest="sec_ch_ua_mobile",
        help="Sec-CH-UA-Mobile header value (?0/?1)",
    )
    parser.add_argument(
        "--sec-ch-ua-platform",
        dest="sec_ch_ua_platform",
        help="Sec-CH-UA-Platform header value (e.g., macOS)",
    )
    parser.add_argument(
        "--sec-ch-ua-platform-version",
        dest="sec_ch_ua_platform_version",
        help="Sec-CH-UA-Platform-Version header value",
    )
    parser.add_argument(
        "--stealth",
        action="store_true",
        help="Enable a maintained stealth preset (webdriver masking, chrome.runtime stub, plugins/mimeTypes, permissions shim)",
    )
    parser.add_argument(
        "--stealth-devtools",
        action="store_true",
        help="Mask devtools/CDP presence even without --stealth (implied by --stealth)",
    )
    parser.add_argument(
        "--no-stealth-devtools",
        action="store_true",
        help="Disable devtools/CDP masking when using --stealth",
    )
    parser.add_argument(
        "--antidetect-preset",
        dest="antidetect_preset",
        choices=sorted(ANTIDETECT_PRESETS.keys()),
        help="Opt-in anti-detect fingerprint preset (e.g., pixelscan)",
    )
    parser.add_argument(
        "--init-script",
        dest="init_script",
        help="Path to a JS file injected with add_init_script before any page scripts run",
    )
    parser.add_argument(
        "--grant-permissions",
        dest="grant_permissions",
        help="Comma-separated permissions to grant to the context (e.g., geolocation,clipboard-read)",
    )
    parser.add_argument(
        "--permissions-origin",
        dest="permissions_origin",
        help="Origin to scope granted permissions to (defaults to all origins)",
    )
    parser.add_argument(
        "--permission-state",
        action="append",
        dest="permission_states",
        default=[],
        help="Override navigator.permissions.query for a permission (format: name=state). Repeatable.",
    )

    args = parser.parse_args()
    preset = _get_antidetect_preset(args.antidetect_preset)

    # Update global configuration
    config.headless = not args.headed
    if preset and "headless" in preset:
        config.headless = preset["headless"]
    config.browser_type = args.browser
    config.timeout = args.timeout
    config.channel = args.channel
    if preset and "channel" in preset:
        config.channel = preset["channel"]
    config.user_data_dir = getattr(args, "user_data_dir", None)
    config.max_elements_returned = args.max_elements
    config.max_element_text_length = args.max_element_text_length
    config.max_accessibility_nodes = args.max_accessibility_nodes
    config.max_response_characters = args.max_response_chars
    config.preview_characters = args.preview_chars
    # Compute per-session artifact directory under the provided base/root.
    artifact_base = Path(args.artifact_dir).expanduser().absolute()
    session_dir = _session_artifact_dir(artifact_base)
    config.artifact_directory = session_dir
    config.artifact_max_age_seconds = max(args.artifact_max_age_seconds, 0)
    config.artifact_max_files = args.artifact_max_files
    config.artifact_chunk_size = max(args.artifact_chunk_size, 256)
    # Apply opt-in preset first so explicit flags can override it.
    _apply_antidetect_preset(preset)

    # User overrides take precedence over preset values.
    if args.user_agent is not None:
        config.user_agent = args.user_agent
    if args.accept_language is not None:
        config.accept_language = args.accept_language
    if args.languages is not None:
        config.languages = _parse_csv_list(args.languages)
    if args.locale is not None:
        config.locale = args.locale
    if args.timezone_id is not None:
        config.timezone_id = args.timezone_id
    if args.device_scale_factor is not None:
        config.device_scale_factor = args.device_scale_factor
    if args.platform is not None:
        config.platform = args.platform
    if args.vendor is not None:
        config.vendor = args.vendor
    if args.hardware_concurrency is not None:
        config.hardware_concurrency = args.hardware_concurrency
    if args.device_memory is not None:
        config.device_memory = args.device_memory
    if args.sec_ch_ua is not None:
        config.sec_ch_ua = args.sec_ch_ua
    if args.sec_ch_ua_mobile is not None:
        config.sec_ch_ua_mobile = args.sec_ch_ua_mobile
    if args.sec_ch_ua_platform is not None:
        config.sec_ch_ua_platform = args.sec_ch_ua_platform
    if args.sec_ch_ua_platform_version is not None:
        config.sec_ch_ua_platform_version = args.sec_ch_ua_platform_version
    if args.sec_ch_ua_full_version_list is not None:
        config.sec_ch_ua_full_version_list = args.sec_ch_ua_full_version_list
    if args.stealth or args.stealth_devtools or args.no_stealth_devtools:
        config.stealth = args.stealth
        config.mask_devtools = (args.stealth and not args.no_stealth_devtools) or args.stealth_devtools
    if args.grant_permissions is not None:
        config.grant_permissions = _parse_csv_list(args.grant_permissions)
    if args.permissions_origin is not None:
        config.permissions_origin = args.permissions_origin
    if args.permission_states:
        config.permission_overrides = {}
        for entry in args.permission_states or []:
            if "=" not in entry:
                logger.warning("Ignoring permission override without '=': %s", entry)
                continue
            name, state = entry.split("=", 1)
            name = name.strip()
            state = state.strip()
            if not name or not state:
                logger.warning("Ignoring malformed permission override: %s", entry)
                continue
            config.permission_overrides[name] = state

    if args.init_script:
        try:
            script_path = Path(args.init_script).expanduser()
            config.custom_init_script = script_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read init script %s: %s", args.init_script, exc)

    # Setup logging before emitting any log lines
    logging.basicConfig(level=logging.INFO)

    # Ensure the session directory exists and apply retention within the session only.
    _enforce_artifact_retention()
    # Prune old session directories across all sessions at startup (TTL-based).
    _enforce_session_retention(artifact_base)

    logger.info("Artifact base: %s", str(artifact_base))
    logger.info("Using per-session artifact dir: %s", str(config.artifact_directory))

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
