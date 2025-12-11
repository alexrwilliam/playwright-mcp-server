import os
import re
import random
from typing import Any, Dict, Tuple

import pytest
from playwright.async_api import async_playwright

from playwright_mcp.server import (
    ANTIDETECT_PRESETS,
    Config,
    _build_context_init_script,
    _build_extra_headers,
)

SUMMARY_RE = re.compile(r"Your results:\s*([A-Za-z ]+)", re.IGNORECASE)
CHECK_RE = re.compile(
    r"(Navigator|Webdriver|CDP|User Agent)\s*Result:\s*(Pass|Fail)\s*([A-Za-z]+)?",
    re.IGNORECASE,
)


def _build_config_from_preset(name: str) -> Config:
    cfg = Config()
    preset = ANTIDETECT_PRESETS.get(name)
    if not preset:
        return cfg
    viewport_pool = preset.get("viewport_pool") or []
    chosen_viewport = random.choice(viewport_pool) if viewport_pool else None
    for key, value in preset.items():
        if key == "viewport_pool":
            continue
        setattr(cfg, key, value)
    if chosen_viewport:
        cfg.viewport_width = chosen_viewport.get("width", cfg.viewport_width)
        cfg.viewport_height = chosen_viewport.get("height", cfg.viewport_height)
        if "device_scale_factor" in chosen_viewport:
            cfg.device_scale_factor = chosen_viewport["device_scale_factor"]
    return cfg


async def _run_pixelscan_check(cfg: Config) -> Tuple[str, Dict[str, Tuple[str, str]], Dict[str, str]]:
    body_text = ""
    async with async_playwright() as p:
        launch_args: Dict[str, Any] = {"headless": cfg.headless}
        if getattr(cfg, "channel", None):
            launch_args["channel"] = cfg.channel
        browser = await p.chromium.launch(**launch_args)
        try:
            headers = _build_extra_headers(cfg)
            context_kwargs: Dict[str, Any] = {}
            if cfg.override_viewport:
                context_kwargs["viewport"] = {
                    "width": cfg.viewport_width,
                    "height": cfg.viewport_height,
                }
                if cfg.device_scale_factor is not None:
                    context_kwargs["device_scale_factor"] = cfg.device_scale_factor
                    context_kwargs.setdefault(
                        "screen",
                        {"width": cfg.viewport_width, "height": cfg.viewport_height},
                    )
            if cfg.user_agent:
                context_kwargs["user_agent"] = cfg.user_agent

            locale_value = cfg.locale or (cfg.languages[0] if cfg.languages else None)
            if locale_value:
                context_kwargs["locale"] = locale_value
            if cfg.timezone_id:
                context_kwargs["timezone_id"] = cfg.timezone_id
            if headers:
                context_kwargs["extra_http_headers"] = headers

            context = await browser.new_context(**context_kwargs)
            init_script = _build_context_init_script(cfg)
            if init_script:
                await context.add_init_script(init_script)
            page = await context.new_page()
            await page.goto("https://pixelscan.net/bot-check", wait_until="networkidle", timeout=30000)
            start_buttons = page.get_by_text("Start", exact=True)
            count = await start_buttons.count()
            if count == 0:
                raise RuntimeError("No Start button found on Pixelscan bot-check page")
            await start_buttons.nth(0).click(timeout=5000)
            await page.wait_for_timeout(4000)
            # Expand technical details if present.
            try:
                details_btn = page.get_by_text("Show technical details")
                if await details_btn.count() > 0:
                    await details_btn.nth(0).click(timeout=2000)
                    await page.wait_for_timeout(500)
            except Exception:
                pass
            # Visit each signal tab to ensure details are rendered.
            tab_text_blobs = []
            for tab_name in ["Navigator", "Webdriver", "CDP", "User Agent"]:
                try:
                    tab = page.get_by_role("button", name=tab_name)
                    if await tab.count() == 0:
                        tab = page.get_by_text(tab_name, exact=True)
                    if await tab.count() > 0:
                        await tab.nth(0).click(timeout=2000)
                        await page.wait_for_timeout(300)
                        tab_text_blobs.append(await page.inner_text("body"))
                except Exception:
                    continue
            if not tab_text_blobs:
                body_text = await page.inner_text("body")
            else:
                body_text = "\n".join(tab_text_blobs)
        finally:
            await browser.close()

    summary_match = SUMMARY_RE.search(body_text)
    summary = summary_match.group(1).strip() if summary_match else ""

    checks: Dict[str, Tuple[str, str]] = {}
    for match in CHECK_RE.finditer(body_text):
        name = match.group(1).strip().title()
        result = match.group(2).strip().title()
        severity = (match.group(3) or "").strip().title()
        checks[name] = (result, severity)

    details: Dict[str, str] = {}
    detail_re = re.compile(r"([A-Za-z][A-Za-z0-9 ]+):\s*(Normal|Suspicious|Detected)", re.IGNORECASE)
    for match in detail_re.finditer(body_text):
        key = match.group(1).strip()
        val = match.group(2).strip().title()
        details[key] = val

    return summary, checks, details


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("PIXELSCAN_RUN"),
    reason="Set PIXELSCAN_RUN=1 to run live pixelscan bot check (networked).",
)
async def test_pixelscan_bot_check():
    preset_name = os.getenv("PIXELSCAN_PRESET", "pixelscan")
    cfg = _build_config_from_preset(preset_name)

    summary, checks, details = await _run_pixelscan_check(cfg)

    print(f"Pixelscan summary: {summary or 'Unknown'}")
    for name, (result, severity) in checks.items():
        print(f"- {name}: {result} ({severity or 'n/a'})")
    if not details:
        print("No detail entries parsed.")
    else:
        suspicious_details = {k: v for k, v in details.items() if v != "Normal"}
        if suspicious_details:
            print("Suspicious details:")
            for k, v in suspicious_details.items():
                print(f"  {k}: {v}")

    assert summary, "Did not find a Pixelscan result summary"
    assert len(checks) >= 3, f"Expected at least 3 signal checks, got {len(checks)}"
