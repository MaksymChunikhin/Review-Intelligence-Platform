"""Small structured-output client for Gemini and Vertex AI."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel
from dotenv import load_dotenv

from src.common.project import find_project_root
StructuredResponse = TypeVar("StructuredResponse", bound=BaseModel)


class GeminiGenerationConfig(Protocol):
    """Structural config shared by discovery, annotation, and analyst calls."""

    provider: str
    backend: str
    model: str
    location: str | None
    temperature: float
    seed: int
    thinking_level: str
    max_output_tokens: int
    max_retries: int
    retry_base_seconds: float


class GeminiConfigurationError(RuntimeError):
    """Report missing local configuration without exposing credentials."""


@dataclass(frozen=True)
class GeminiResponseMetadata:
    """Preserve provider metadata needed for cost and lineage reports."""

    response_id: str | None
    model_version: str | None
    usage_metadata: dict[str, Any]


@dataclass(frozen=True)
class GeminiEnvironmentStatus:
    """Report non-secret local readiness for the configured backend."""

    provider: str
    backend: str
    model: str
    location: str | None
    sdk_installed: bool
    sdk_version: str | None
    cloud_project_configured: bool
    local_adc_hint_present: bool
    api_key_configured: bool
    ready_for_client_initialization: bool
    missing_requirements: list[str]


def load_project_environment() -> None:
    """Load an ignored project `.env` without overriding shell settings."""
    try:
        root = find_project_root(Path.cwd())
    except FileNotFoundError:
        return
    load_dotenv(root / ".env", override=False)


def inspect_gemini_environment(
    config: GeminiGenerationConfig,
) -> GeminiEnvironmentStatus:
    """Inspect required settings without reading or displaying credentials."""
    load_project_environment()
    try:
        from google import genai

        sdk_installed = True
        sdk_version = getattr(genai, "__version__", None)
    except ImportError:
        sdk_installed = False
        sdk_version = None
    cloud_project_configured = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    explicit_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    default_adc_path = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    local_adc_hint_present = bool(
        (explicit_credentials and Path(explicit_credentials).is_file())
        or default_adc_path.is_file()
    )
    api_key_configured = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    missing: list[str] = []
    if not sdk_installed:
        missing.append("google-genai package")
    if config.backend == "vertex_ai" and not cloud_project_configured:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if config.backend == "vertex_ai" and not local_adc_hint_present:
        missing.append("Application Default Credentials")
    if config.backend == "gemini_api" and not api_key_configured:
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY")
    return GeminiEnvironmentStatus(
        provider=config.provider,
        backend=config.backend,
        model=config.model,
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", config.location),
        sdk_installed=sdk_installed,
        sdk_version=sdk_version,
        cloud_project_configured=cloud_project_configured,
        local_adc_hint_present=local_adc_hint_present,
        api_key_configured=api_key_configured,
        ready_for_client_initialization=not missing,
        missing_requirements=missing,
    )


def create_genai_client(config: GeminiGenerationConfig) -> Any:
    """Create a Google Gen AI client from environment-owned credentials."""
    load_project_environment()
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise GeminiConfigurationError(
            "google-genai is not installed; install project requirements"
        ) from error

    if config.backend == "vertex_ai":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise GeminiConfigurationError(
                "GOOGLE_CLOUD_PROJECT is required for the Vertex AI backend"
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", config.location)
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(
                timeout=int(1000 * getattr(config, "request_timeout_seconds", 45))
            ),
        )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini API"
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(1000 * getattr(config, "request_timeout_seconds", 45))
        ),
    )


class GeminiStructuredClient:
    """Generate validated Pydantic responses with bounded retries."""

    def __init__(
        self,
        config: GeminiGenerationConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client or create_genai_client(config)

    def generate(
        self,
        *,
        contents: str,
        system_instruction: str,
        response_model: type[StructuredResponse],
    ) -> tuple[StructuredResponse, GeminiResponseMetadata, str]:
        """Call Gemini and validate its structured response."""
        from google.genai import types

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_json_schema=response_model.model_json_schema(),
                        temperature=self.config.temperature,
                        seed=self.config.seed,
                        max_output_tokens=self.config.max_output_tokens,
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(disable=True)
                        ),
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self.config.thinking_level
                        ),
                    ),
                )
                text = response.text or ""
                if response.parsed is not None:
                    parsed = response_model.model_validate(response.parsed)
                else:
                    parsed = response_model.model_validate_json(text)
                usage = (
                    response.usage_metadata.model_dump(
                        mode="json", exclude_none=True
                    )
                    if response.usage_metadata is not None
                    else {}
                )
                metadata = GeminiResponseMetadata(
                    response_id=response.response_id,
                    model_version=response.model_version,
                    usage_metadata=usage,
                )
                return parsed, metadata, text
            except Exception as error:  # provider and validation exceptions
                last_error = error
                if getattr(error, "code", None) in {400, 401, 403, 404}:
                    break
                if attempt + 1 < self.config.max_retries:
                    time.sleep(
                        min(self.config.retry_base_seconds * 2**attempt, 30)
                    )
        raise RuntimeError(
            f"Gemini structured generation failed after "
            f"{self.config.max_retries} attempts"
        ) from last_error

    def close(self) -> None:
        """Release SDK network resources."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


def main() -> None:
    """Report non-secret readiness for one discovery configuration."""
    from src.schemas.aspects import load_aspect_discovery_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    status = inspect_gemini_environment(
        load_aspect_discovery_config(args.config_path)
    )
    payload = asdict(status)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(args.output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
