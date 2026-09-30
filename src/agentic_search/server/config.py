"""Service configuration: auth, CORS, capacity, and named harness profiles with limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_search.config import ConfigError, build_harness
from agentic_search.core.harness import Harness
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import Budget


class AuthConfig(BaseModel):
    """`api_key`: clients send `Authorization: Bearer <key>` or `X-API-Key: <key>`; valid keys are
    read from the comma-separated environment variable `keys_env`. `none` must be explicit."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "api_key"]
    keys_env: str | None = None

    @model_validator(mode="after")
    def _keys_env_for_api_key(self) -> AuthConfig:
        if self.type == "api_key" and not self.keys_env:
            raise ValueError("auth type api_key needs keys_env")
        return self

    def keys(self) -> list[str]:
        if self.type == "none":
            return []
        raw = os.environ.get(self.keys_env or "", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ConfigError(f"environment variable {self.keys_env!r} holds no API keys")
        for k in keys:
            register_secret(k)
        return keys


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth: AuthConfig
    cors_origins: list[str] = Field(default_factory=list)
    max_concurrent_searches: int = Field(default=16, ge=1)
    default_profile: str | None = None
    keepalive_s: float = Field(default=15.0, gt=0)
    max_image_bytes: int = Field(default=10_000_000, ge=1)
    max_images: int = Field(default=4, ge=0)
    max_body_bytes: int | None = Field(default=None, ge=1)

    def body_limit(self) -> int:
        """`max_body_bytes`, or by default room for `max_images` base64 images plus 64 KiB."""
        if self.max_body_bytes is not None:
            return self.max_body_bytes
        return self.max_images * self.max_image_bytes * 4 // 3 + 65_536


class ProfileLimits(BaseModel):
    """`max_budget`: per-field ceilings on the budget a request may ask for (unset = no ceiling)."""

    model_config = ConfigDict(extra="forbid")

    max_budget: dict[str, float | int] = Field(default_factory=dict)
    allow_include_content: bool = True

    @model_validator(mode="after")
    def _known_budget_fields(self) -> ProfileLimits:
        unknown = set(self.max_budget) - set(Budget.model_fields)
        if unknown:
            raise ValueError(f"unknown max_budget fields: {sorted(unknown)}")
        Budget(**self.max_budget)  # ceilings must themselves be a valid budget
        if any(v <= 0 for v in self.max_budget.values()):
            raise ValueError("max_budget ceilings must be positive")
        return self


@dataclass
class Profile:
    harness: Harness
    limits: ProfileLimits = field(default_factory=ProfileLimits)


def load_service(path: str | Path) -> tuple[ServiceConfig, dict[str, Profile]]:
    """Read a service YAML: a `service:` block and `profiles:` of harness configs, each with an
    optional `limits:` block. Every profile's harness is built here (not yet set up)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw.get("profiles"), dict) or not raw["profiles"]:
        raise ConfigError("service config needs a non-empty `profiles:` mapping")
    try:
        config = ServiceConfig(**(raw.get("service") or {}))
    except ValueError as exc:
        raise ConfigError(f"invalid `service:` block: {exc}") from exc
    profiles: dict[str, Profile] = {}
    for name, cfg in raw["profiles"].items():
        cfg = dict(cfg or {})
        try:
            limits = ProfileLimits(**(cfg.pop("limits", None) or {}))
        except ValueError as exc:
            raise ConfigError(f"profile {name!r} has invalid limits: {exc}") from exc
        profiles[str(name)] = Profile(harness=build_harness(cfg, base_dir=path.parent), limits=limits)
    return config, profiles


def check_profiles(config: ServiceConfig, profiles: dict[str, Any]) -> None:
    if not profiles:
        raise ConfigError("the service needs at least one profile")
    if config.default_profile is not None and config.default_profile not in profiles:
        raise ConfigError(f"default_profile {config.default_profile!r} is not a profile")
