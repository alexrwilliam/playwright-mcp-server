from playwright_mcp.server import (
    _apply_cli_stealth_overrides,
    Config,
    _apply_automation_flag_hardening,
    _apply_default_headless_hardening,
    _compute_headless_hardening_forces,
    _default_headless_mode,
)


def test_default_config_uses_cli_style_hardening_baseline():
    cfg = Config()

    assert cfg.browser_type == "chromium"
    assert cfg.headless == _default_headless_mode()
    assert cfg.channel == "chrome"
    if cfg.headless:
        assert cfg.stealth is True
        assert cfg.mask_devtools is True
    else:
        assert cfg.stealth is False
    assert cfg.disable_blink_automation is True
    assert cfg.override_viewport is False


def test_non_chromium_default_channel_is_none():
    cfg = Config(browser_type="firefox")
    assert cfg.channel is None


def test_automation_flag_hardening_is_idempotent():
    launch_options = {"headless": False}

    _apply_automation_flag_hardening(launch_options)
    _apply_automation_flag_hardening(launch_options)

    assert launch_options["ignore_default_args"].count("--enable-automation") == 1
    assert (
        launch_options["args"].count(
            "--disable-blink-features=AutomationControlled"
        )
        == 1
    )


def test_headless_mode_defaults_to_hardened_stealth_profile():
    cfg = Config(headless=True, stealth=False, mask_devtools=False)

    assert cfg.stealth is True
    assert cfg.mask_devtools is True
    assert cfg.disable_blink_automation is True
    assert cfg.channel == "chrome"


def test_headless_hardening_can_respect_explicit_runtime_overrides():
    cfg = Config(headless=False, stealth=False, mask_devtools=False)
    cfg.headless = True

    _apply_default_headless_hardening(
        cfg,
        force_stealth=False,
        force_mask_devtools=False,
    )

    assert cfg.stealth is False
    assert cfg.mask_devtools is False
    assert cfg.disable_blink_automation is True


def test_strict_preset_disables_headless_forcing_for_preset_keys():
    preset = {
        "stealth": False,
        "mask_devtools": False,
        "disable_blink_automation": False,
    }
    force_stealth, force_mask, force_disable_blink = _compute_headless_hardening_forces(
        explicit_stealth=False,
        explicit_mask_devtools=False,
        preset=preset,
        strict_preset=True,
    )

    assert force_stealth is False
    assert force_mask is False
    assert force_disable_blink is False


def test_non_strict_preset_keeps_headless_forcing_enabled():
    preset = {
        "stealth": False,
        "mask_devtools": False,
        "disable_blink_automation": False,
    }
    force_stealth, force_mask, force_disable_blink = _compute_headless_hardening_forces(
        explicit_stealth=False,
        explicit_mask_devtools=False,
        preset=preset,
        strict_preset=False,
    )

    assert force_stealth is True
    assert force_mask is True
    assert force_disable_blink is True


def test_cli_no_stealth_devtools_does_not_disable_existing_stealth():
    cfg = Config(headless=True, stealth=True, mask_devtools=True)

    explicit_stealth, explicit_mask = _apply_cli_stealth_overrides(
        cfg,
        stealth=False,
        stealth_devtools=False,
        no_stealth_devtools=True,
    )

    assert explicit_stealth is False
    assert explicit_mask is True
    assert cfg.stealth is True
    assert cfg.mask_devtools is False
