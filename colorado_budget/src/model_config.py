"""
Model + prompt configuration for the Colorado Budget Agent.

Reads config.toml, resolves a named profile, and builds the matching Strands
model provider. This lets you swap LLMs (Claude, Devstral, NVIDIA NIM, any
OpenAI-compatible endpoint) without touching agent code.

The config file holds the system prompt and, per profile, which provider /
model / API-key-env to use. API keys themselves are NEVER stored in the config —
each profile names an `api_key_env` and the key is read from the environment
(or a .env file loaded by utils.load_api_keys).
"""
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


@dataclass
class AgentConfig:
    profile_name: str
    provider: str
    model_id: str
    system_prompt: str
    model: Any  # a Strands Model instance


def _read_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _require_key(api_key_env: str, profile_name: str) -> str:
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(
            f"Profile '{profile_name}' requires environment variable ${api_key_env}, "
            f"but it is not set. Add it to colorado_budget/.env or your shell environment."
        )
    return key


def _build_model(profile: dict, profile_name: str):
    """Construct a Strands model provider from a profile dict."""
    provider = str(profile.get("provider", "")).lower()
    model_id = profile["model_id"]
    max_tokens = int(profile.get("max_tokens", 8096))
    params: dict = dict(profile.get("params") or {})
    api_key_env: Optional[str] = profile.get("api_key_env")

    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        key = _require_key(api_key_env or "ANTHROPIC_API_KEY", profile_name)
        return AnthropicModel(
            client_args={"api_key": key},
            model_id=model_id,
            max_tokens=max_tokens,
            params=params or None,
        )

    if provider == "mistral":
        from strands.models.mistral import MistralModel

        key = _require_key(api_key_env or "MISTRAL_API_KEY", profile_name)
        # MistralModel takes temperature/top_p as direct config keys, not a
        # nested `params` dict — so flatten them in.
        return MistralModel(
            api_key=key,
            model_id=model_id,
            max_tokens=max_tokens,
            **params,
        )

    if provider == "openai":
        # Covers OpenAI and any OpenAI-compatible gateway (NVIDIA build.nvidia.com,
        # OpenRouter, local vLLM/Ollama, ...). base_url selects the endpoint.
        from strands.models.openai import OpenAIModel

        key = _require_key(api_key_env or "OPENAI_API_KEY", profile_name)
        client_args: dict = {"api_key": key}
        base_url = profile.get("base_url")
        if base_url:
            client_args["base_url"] = base_url
        params.setdefault("max_tokens", max_tokens)
        return OpenAIModel(
            client_args=client_args,
            model_id=model_id,
            params=params,
        )

    raise ValueError(
        f"Unknown provider '{provider}' in profile '{profile_name}'. "
        f"Supported: anthropic, mistral, openai."
    )


def load_agent_config(
    profile_name: Optional[str] = None,
    config_path: Path = CONFIG_PATH,
) -> AgentConfig:
    """Load config.toml, resolve the active profile, and build its model.

    Args:
        profile_name: override the config's `active_profile`. Falls back to the
            config default when None.
        config_path: path to the TOML config file.
    """
    cfg = _read_config(config_path)
    profiles = cfg.get("profiles", {})
    if not profiles:
        raise ValueError(f"No [profiles.*] defined in {config_path}")

    name = profile_name or cfg.get("active_profile")
    if not name:
        raise ValueError(
            "No profile selected: pass --profile or set active_profile in config.toml"
        )
    if name not in profiles:
        raise ValueError(
            f"Profile '{name}' not found in {config_path}. "
            f"Available: {', '.join(sorted(profiles))}"
        )

    profile = profiles[name]

    # Per-profile system_prompt overrides the shared top-level default.
    system_prompt = profile.get("system_prompt") or cfg.get("system_prompt")
    if not system_prompt:
        raise ValueError(
            "No system_prompt found (neither top-level nor in the selected profile)."
        )

    model = _build_model(profile, name)
    logger.info(
        f"Profile '{name}': provider={profile.get('provider')} model={profile['model_id']}"
    )
    return AgentConfig(
        profile_name=name,
        provider=str(profile.get("provider", "")),
        model_id=profile["model_id"],
        system_prompt=system_prompt,
        model=model,
    )


def list_profiles(config_path: Path = CONFIG_PATH) -> list[str]:
    cfg = _read_config(config_path)
    return sorted(cfg.get("profiles", {}).keys())
