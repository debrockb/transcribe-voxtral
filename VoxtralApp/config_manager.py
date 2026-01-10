"""
Configuration Manager for Voxtral App
Handles reading and updating application configuration
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default configuration file path
CONFIG_FILE = Path(__file__).parent / "config.json"


class ConfigManager:
    """Manages application configuration"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (defaults to config.json)
        """
        self.config_path = config_path or CONFIG_FILE
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model": {
                "version": "full",
                "available_models": {
                    "full": {
                        "id": "mistralai/Voxtral-Mini-3B-2507",
                        "name": "Voxtral Mini 3B (Full)",
                        "size_gb": 9.36,
                        "format": "safetensors",
                        "quantization": None,
                        "description": "Full precision model - Best quality, requires ~20-30GB RAM during loading (CPU loading: 10-30 min)",
                        "memory_requirements": {
                            "disk": "9.36 GB",
                            "ram_loading": "20-30 GB (Windows: increase pagefile to 50GB)",
                            "ram_inference": "10-12 GB",
                            "load_time_cpu": "10-30 minutes (use NVIDIA GPU for faster loading)",
                        },
                    },
                    "quantized": {
                        "id": "mistralai/Voxtral-Mini-3B-2507",
                        "name": "Voxtral Mini 3B (4-bit Quantized)",
                        "size_gb": 9.36,
                        "format": "safetensors",
                        "quantization": "4bit",
                        "description": "4-bit quantized model - 75% less memory usage (NVIDIA GPU required, auto-falls back to full precision on Mac/CPU)",
                        "memory_requirements": {
                            "disk": "9.36 GB (quantized during load)",
                            "ram_loading": "5-8 GB (NVIDIA GPU) / 20-30GB (Mac/CPU fallback)",
                            "ram_inference": "3-4 GB (NVIDIA GPU) / 10-12GB (Mac/CPU fallback)",
                        },
                    },
                    "small-24b-full": {
                        "id": "mistralai/Voxtral-Small-24B-2507",
                        "name": "Voxtral Small 24B (Full)",
                        "size_gb": 48.0,
                        "format": "safetensors",
                        "quantization": None,
                        "description": "Large 24B parameter model - Superior accuracy for complex audio, requires high-end hardware",
                        "memory_requirements": {
                            "disk": "48 GB",
                            "ram_loading": "80-100 GB (Windows: increase pagefile to 120GB)",
                            "ram_inference": "50-60 GB",
                            "load_time_cpu": "30-60 minutes (NVIDIA GPU strongly recommended)",
                        },
                    },
                    "small-24b-quantized": {
                        "id": "mistralai/Voxtral-Small-24B-2507",
                        "name": "Voxtral Small 24B (4-bit Quantized)",
                        "size_gb": 48.0,
                        "format": "safetensors",
                        "quantization": "4bit",
                        "description": "4-bit quantized 24B model - Best accuracy with reduced memory (NVIDIA GPU required)",
                        "memory_requirements": {
                            "disk": "48 GB (quantized during load)",
                            "ram_loading": "20-30 GB (NVIDIA GPU)",
                            "ram_inference": "15-20 GB (NVIDIA GPU)",
                            "load_time_cpu": "N/A - NVIDIA GPU required for quantization",
                        },
                    },
                },
            },
            "app": {"version": "1.1.0", "name": "Voxtral Transcription"},
        }

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path (e.g., "model.version")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path (e.g., "model.version")
            value: Value to set
        """
        keys = key_path.split(".")
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def get_model_config(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for specified model version.

        Args:
            model_version: Model version key ('full' or 'quantized'), uses current if None

        Returns:
            Model configuration dict
        """
        if model_version is None:
            model_version = self.get("model.version", "full")

        available_models = self.get("model.available_models", {})
        return available_models.get(model_version, available_models.get("full"))

    def set_model_version(self, version: str):
        """
        Set the active model version.

        Args:
            version: Model version key ('full' or 'quantized')
        """
        available = self.get("model.available_models", {})
        if version not in available:
            raise ValueError(f"Invalid model version: {version}. Available: {list(available.keys())}")

        self.set("model.version", version)
        self.save_config()
        logger.info(f"Model version set to: {version}")

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models configuration."""
        return self.get("model.available_models", {})

    def get_current_model_id(self) -> str:
        """Get the Hugging Face model ID for the current model."""
        model_config = self.get_model_config()
        return model_config.get("id", "mistralai/Voxtral-Mini-3B-2507")

    def get_app_version(self) -> str:
        """Get application version."""
        return self.get("app.version", "1.0.0")

    def get_app_name(self) -> str:
        """Get application name."""
        return self.get("app.name", "Voxtral Transcription")


# Global configuration instance
config = ConfigManager()
