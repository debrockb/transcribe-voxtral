"""
Tests for update_checker module
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

import update_checker


@pytest.mark.unit
class TestGetCurrentVersion:
    """Test get_current_version function"""

    def test_get_version_file_exists(self, tmp_path):
        """Test reading version from existing VERSION file"""
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.2.3")

        with patch.object(update_checker, "VERSION_FILE", version_file):
            version = update_checker.get_current_version()
            assert version == "1.2.3"

    def test_get_version_file_missing(self, tmp_path):
        """Test reading version when VERSION file doesn't exist"""
        version_file = tmp_path / "VERSION"

        with patch.object(update_checker, "VERSION_FILE", version_file):
            version = update_checker.get_current_version()
            assert version == "0.0.0"

    def test_get_version_file_error(self, tmp_path):
        """Test reading version when file read fails"""
        version_file = tmp_path / "VERSION"

        with patch.object(update_checker, "VERSION_FILE", version_file):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", side_effect=IOError("Read error")):
                    version = update_checker.get_current_version()
                    assert version == "0.0.0"

    def test_get_version_strips_whitespace(self, tmp_path):
        """Test version string is properly stripped of whitespace"""
        version_file = tmp_path / "VERSION"
        version_file.write_text("  1.2.3  \n")

        with patch.object(update_checker, "VERSION_FILE", version_file):
            version = update_checker.get_current_version()
            assert version == "1.2.3"


@pytest.mark.unit
class TestGetLatestRelease:
    """Test get_latest_release function"""

    @patch("update_checker.requests.get")
    def test_successful_release_fetch(self, mock_get):
        """Test successfully fetching latest release from GitHub"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v2.0.0",
            "name": "Version 2.0.0",
            "body": "Release notes here",
            "published_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/debrockb/transcribe-voxtral/releases/tag/v2.0.0",
            "zipball_url": "https://github.com/debrockb/transcribe-voxtral/archive/v2.0.0.zip",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = update_checker.get_latest_release()

        assert result is not None
        assert result["version"] == "2.0.0"  # v prefix stripped
        assert result["name"] == "Version 2.0.0"
        assert result["body"] == "Release notes here"
        assert result["html_url"] == "https://github.com/debrockb/transcribe-voxtral/releases/tag/v2.0.0"

    @patch("update_checker.requests.get")
    def test_release_fetch_timeout(self, mock_get):
        """Test handling of timeout when fetching release"""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = update_checker.get_latest_release()

        assert result is None

    @patch("update_checker.requests.get")
    def test_release_fetch_network_error(self, mock_get):
        """Test handling of network errors"""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = update_checker.get_latest_release()

        assert result is None

    @patch("update_checker.requests.get")
    def test_release_fetch_404(self, mock_get):
        """Test handling of 404 (no releases exist)"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        result = update_checker.get_latest_release()

        assert result is None

    @patch("update_checker.requests.get")
    def test_release_fetch_invalid_json(self, mock_get):
        """Test handling of invalid JSON response"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = update_checker.get_latest_release()

        assert result is None

    @patch("update_checker.requests.get")
    def test_release_fetch_missing_fields(self, mock_get):
        """Test handling of response with missing required fields"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v2.0.0",
            # Missing other required fields
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = update_checker.get_latest_release()

        # Should handle missing fields gracefully
        assert result is None


@pytest.mark.unit
class TestCheckForUpdates:
    """Test check_for_updates function"""

    @patch("update_checker.get_current_version")
    @patch("update_checker.get_latest_release")
    def test_update_available_newer_version(self, mock_latest, mock_current):
        """Test when newer version is available"""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = {
            "version": "2.0.0",
            "name": "v2.0.0",
            "body": "New features",
            "published_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/test/releases/tag/v2.0.0",
            "download_url": "https://github.com/test/archive/v2.0.0.zip",
        }

        result = update_checker.check_for_updates()

        assert result["update_available"] is True
        assert result["current_version"] == "1.0.0"
        assert result["latest_version"] == "2.0.0"
        assert "release_name" in result
        assert "release_notes" in result

    @patch("update_checker.get_current_version")
    @patch("update_checker.get_latest_release")
    def test_no_update_same_version(self, mock_latest, mock_current):
        """Test when same version is installed"""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = {
            "version": "1.0.0",
            "name": "v1.0.0",
            "body": "Current version",
            "published_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/test/releases/tag/v1.0.0",
            "download_url": "https://github.com/test/archive/v1.0.0.zip",
        }

        result = update_checker.check_for_updates()

        assert result["update_available"] is False
        assert result["current_version"] == "1.0.0"
        assert result["latest_version"] == "1.0.0"

    @patch("update_checker.get_current_version")
    @patch("update_checker.get_latest_release")
    def test_no_update_older_version_available(self, mock_latest, mock_current):
        """Test when installed version is newer than latest release"""
        mock_current.return_value = "2.0.0"
        mock_latest.return_value = {
            "version": "1.0.0",
            "name": "v1.0.0",
            "body": "Old version",
            "published_at": "2023-01-01T00:00:00Z",
            "html_url": "https://github.com/test/releases/tag/v1.0.0",
            "download_url": "https://github.com/test/archive/v1.0.0.zip",
        }

        result = update_checker.check_for_updates()

        assert result["update_available"] is False

    @patch("update_checker.get_current_version")
    @patch("update_checker.get_latest_release")
    def test_update_check_network_failure(self, mock_latest, mock_current):
        """Test handling of network failure when checking for updates"""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = None  # Network error

        result = update_checker.check_for_updates()

        assert result["update_available"] is False
        assert "error" in result
        assert "current_version" in result

    @patch("update_checker.get_current_version")
    @patch("update_checker.get_latest_release")
    def test_version_comparison_with_patch(self, mock_latest, mock_current):
        """Test semantic version comparison with patch versions"""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = {
            "version": "1.0.1",
            "name": "v1.0.1",
            "body": "Patch release",
            "published_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/test/releases/tag/v1.0.1",
            "download_url": "https://github.com/test/archive/v1.0.1.zip",
        }

        result = update_checker.check_for_updates()

        assert result["update_available"] is True
        assert result["latest_version"] == "1.0.1"

    @patch("update_checker.get_current_version")
    @patch("update_checker.get_latest_release")
    def test_version_comparison_with_prerelease(self, mock_latest, mock_current):
        """Test version comparison handles prereleases"""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = {
            "version": "2.0.0-beta.1",
            "name": "v2.0.0-beta.1",
            "body": "Beta release",
            "published_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/test/releases/tag/v2.0.0-beta.1",
            "download_url": "https://github.com/test/archive/v2.0.0-beta.1.zip",
        }

        result = update_checker.check_for_updates()

        # Prerelease should be considered newer
        assert result["update_available"] is True


@pytest.mark.unit
class TestUpdateCheckerConstants:
    """Test module constants are properly configured"""

    def test_github_repo_configured(self):
        """Test GitHub repository is configured"""
        assert update_checker.GITHUB_REPO == "debrockb/transcribe-voxtral"

    def test_github_api_url_configured(self):
        """Test GitHub API URL is properly formatted"""
        assert "api.github.com" in update_checker.GITHUB_API_URL
        assert "debrockb/transcribe-voxtral" in update_checker.GITHUB_API_URL
        assert "releases/latest" in update_checker.GITHUB_API_URL

    def test_version_file_path_configured(self):
        """Test VERSION file path is properly configured"""
        assert update_checker.VERSION_FILE.name == "VERSION"
