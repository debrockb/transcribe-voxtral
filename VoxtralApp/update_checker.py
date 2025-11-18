"""
Update Checker Module
Checks GitHub for new releases and manages version information
"""

import logging
from pathlib import Path

import requests
from packaging import version

logger = logging.getLogger(__name__)

# GitHub repository configuration
GITHUB_REPO = "debrockb/transcribe-voxtral"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
VERSION_FILE = Path(__file__).parent / "VERSION"


def get_current_version():
    """
    Read current version from VERSION file.

    Returns:
        str: Current version number (e.g., "1.0.0")
    """
    try:
        if VERSION_FILE.exists():
            return VERSION_FILE.read_text().strip()
        logger.warning("VERSION file not found, defaulting to 0.0.0")
        return "0.0.0"
    except Exception as e:
        logger.error(f"Error reading version file: {e}")
        return "0.0.0"


def get_latest_release():
    """
    Fetch latest release information from GitHub.

    Returns:
        dict: Release information or None if fetch fails
    """
    try:
        response = requests.get(GITHUB_API_URL, timeout=5)
        response.raise_for_status()

        data = response.json()
        return {
            "version": data["tag_name"].lstrip("v"),
            "name": data["name"],
            "body": data["body"],
            "published_at": data["published_at"],
            "html_url": data["html_url"],
            "download_url": data["zipball_url"],
        }
    except requests.exceptions.Timeout:
        logger.error("Timeout while fetching latest release")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching latest release: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing release data: {e}")
        return None


def check_for_updates():
    """
    Check if a new version is available on GitHub.

    Returns:
        dict: Update information with keys:
            - update_available (bool): Whether an update is available
            - current_version (str): Current installed version
            - latest_version (str): Latest available version
            - release_name (str): Name of the latest release
            - release_notes (str): Release notes/changelog
            - release_url (str): URL to GitHub release page
            - download_url (str): URL to download the release
            - published_at (str): ISO timestamp of release
            - error (str): Error message if check failed
    """
    current = get_current_version()
    latest = get_latest_release()

    if not latest:
        return {
            "update_available": False,
            "current_version": current,
            "error": "Could not check for updates. Please check your internet connection.",
        }

    latest_version = latest["version"]

    try:
        # Compare versions using packaging library
        is_newer = version.parse(latest_version) > version.parse(current)
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        is_newer = False

    return {
        "update_available": is_newer,
        "current_version": current,
        "latest_version": latest_version,
        "release_name": latest.get("name"),
        "release_notes": latest.get("body"),
        "release_url": latest.get("html_url"),
        "download_url": latest.get("download_url"),
        "published_at": latest.get("published_at"),
    }


if __name__ == "__main__":
    # Test the update checker
    logging.basicConfig(level=logging.INFO)
    print(f"Current version: {get_current_version()}")
    print("\nChecking for updates...")
    update_info = check_for_updates()
    print(f"\nUpdate available: {update_info['update_available']}")
    if update_info.get("update_available"):
        print(f"Latest version: {update_info['latest_version']}")
        print(f"Release: {update_info['release_name']}")
