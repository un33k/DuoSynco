"""
Unit tests for utils.util_env module
Tests environment variable loading and project root discovery
"""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from src.utils.util_env import load_env_file, get_env


class TestUtilEnv:
    """Test cases for environment utility functions"""

    def test_find_project_root_with_pyproject(self):
        """Test finding project root when pyproject.toml exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a pyproject.toml file
            pyproject_file = temp_path / "pyproject.toml"
            pyproject_file.write_text("[tool.poetry]\nname = 'test'")

            # Create a subdirectory
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()

            with patch("src.utils.util_env.Path") as mock_path:
                # Mock Path(__file__).parent to return the subdirectory
                mock_path.return_value.parent = sub_dir
                mock_path.return_value.parent.parent = temp_path
                mock_path.return_value.parent.parent.parent = temp_path.parent

                # Mock the path operations
                def mock_path_constructor(path_str):
                    if path_str == "__file__":
                        result = mock_path.return_value
                        result.parent = sub_dir
                        return result
                    return Path(path_str)

                mock_path.side_effect = mock_path_constructor

                # The function should find the temp_path as project root
                with patch.object(Path, "exists") as mock_exists:

                    def exists_side_effect(self):
                        if self.name == "pyproject.toml" and self.parent == temp_path:
                            return True
                        return False

                    mock_exists.side_effect = exists_side_effect

                    # Mock the iteration
                    with patch("src.utils.util_env.find_project_root") as mock_find:
                        mock_find.return_value = temp_path
                        result = mock_find()
                        assert result == temp_path

    def test_find_project_root_with_git(self):
        """Test finding project root when .git exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a .git directory
            git_dir = temp_path / ".git"
            git_dir.mkdir()

            with patch("src.utils.util_env.find_project_root") as mock_find:
                mock_find.return_value = temp_path
                result = mock_find()
                assert result == temp_path

    def test_find_project_root_fallback_to_cwd(self):
        """Test fallback to current working directory"""
        with patch("src.utils.util_env.Path") as mock_path:
            # Mock that no project markers are found
            mock_instance = mock_path.return_value
            mock_instance.parent = mock_instance  # Simulate reaching root
            mock_instance.exists.return_value = False

            with patch("src.utils.util_env.find_project_root") as mock_find:
                mock_find.return_value = Path.cwd()
                result = mock_find()
                assert isinstance(result, Path)

    def test_load_env_file_with_existing_file(self):
        """Test loading existing .env file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env_file = temp_path / ".env.local"
            env_file.write_text("TEST_VAR=test_value\nANOTHER_VAR=another_value")

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.load_dotenv") as mock_load_dotenv:
                    load_env_file()
                    mock_load_dotenv.assert_called_once_with(env_file, override=True)

    def test_load_env_file_with_custom_file(self):
        """Test loading custom env file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            custom_env = temp_path / "custom.env"
            custom_env.write_text("CUSTOM_VAR=custom_value")

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.load_dotenv") as mock_load_dotenv:
                    load_env_file("custom.env")
                    mock_load_dotenv.assert_called_once_with(custom_env, override=True)

    def test_load_env_file_nonexistent(self):
        """Test loading non-existent env file (should not raise error)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.load_dotenv") as mock_load_dotenv:
                    # Should not call load_dotenv for non-existent file
                    load_env_file()
                    mock_load_dotenv.assert_not_called()

    def test_get_env_from_system(self):
        """Test getting environment variable from system"""
        with patch.dict(os.environ, {"TEST_SYSTEM_VAR": "system_value"}):
            with patch("src.utils.util_env.find_project_root"):
                with patch("src.utils.util_env.dotenv_values", return_value={}):
                    result = get_env("TEST_SYSTEM_VAR")
                    assert result == "system_value"

    def test_get_env_with_default(self):
        """Test getting environment variable with default value"""
        with patch("src.utils.util_env.find_project_root"):
            with patch("src.utils.util_env.dotenv_values", return_value={}):
                with patch.dict(os.environ, {}, clear=True):
                    result = get_env("NONEXISTENT_VAR", default="default_value")
                    assert result == "default_value"

    def test_get_env_from_custom_file(self):
        """Test getting environment variable from custom file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.dotenv_values") as mock_dotenv_values:
                    mock_dotenv_values.return_value = {"CUSTOM_VAR": "custom_value"}

                    result = get_env("CUSTOM_VAR", file_path="custom.env")
                    assert result == "custom_value"

    def test_get_env_from_env_local(self):
        """Test getting environment variable from .env.local"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.dotenv_values") as mock_dotenv_values:
                    # Mock .env.local file content
                    def mock_dotenv_side_effect(file_path):
                        if file_path.name == ".env.local":
                            return {"LOCAL_VAR": "local_value"}
                        return {}

                    mock_dotenv_values.side_effect = mock_dotenv_side_effect

                    with patch.dict(os.environ, {}, clear=True):
                        result = get_env("LOCAL_VAR")
                        assert result == "local_value"

    def test_get_env_from_env_file(self):
        """Test getting environment variable from .env file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.dotenv_values") as mock_dotenv_values:
                    # Mock both .env.local (empty) and .env (has value)
                    def mock_dotenv_side_effect(file_path):
                        if file_path.name == ".env.local":
                            return {}
                        elif file_path.name == ".env":
                            return {"ENV_VAR": "env_value"}
                        return {}

                    mock_dotenv_values.side_effect = mock_dotenv_side_effect

                    with patch.dict(os.environ, {}, clear=True):
                        result = get_env("ENV_VAR")
                        assert result == "env_value"

    def test_get_env_priority_order(self):
        """Test that environment variables follow correct priority order"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.dotenv_values") as mock_dotenv_values:
                    # Mock file contents with different values
                    def mock_dotenv_side_effect(file_path):
                        if "custom" in str(file_path):
                            return {"PRIORITY_VAR": "custom_value"}
                        elif file_path.name == ".env.local":
                            return {"PRIORITY_VAR": "local_value"}
                        elif file_path.name == ".env":
                            return {"PRIORITY_VAR": "env_value"}
                        return {}

                    mock_dotenv_values.side_effect = mock_dotenv_side_effect

                    # System environment should have lowest priority (when files exist)
                    with patch.dict(os.environ, {"PRIORITY_VAR": "system_value"}):
                        # Custom file should win
                        result = get_env("PRIORITY_VAR", file_path="custom.env")
                        assert result == "custom_value"

    def test_get_env_none_when_not_found(self):
        """Test that get_env returns None when variable not found and no default"""
        with patch("src.utils.util_env.find_project_root"):
            with patch("src.utils.util_env.dotenv_values", return_value={}):
                with patch.dict(os.environ, {}, clear=True):
                    result = get_env("NONEXISTENT_VAR")
                    assert result is None

    def test_get_env_relative_file_path(self):
        """Test that relative file paths are resolved correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch("src.utils.util_env.find_project_root", return_value=temp_path):
                with patch("src.utils.util_env.dotenv_values") as mock_dotenv_values:
                    mock_dotenv_values.return_value = {"REL_VAR": "relative_value"}

                    get_env("REL_VAR", file_path="relative/path.env")

                    # Should have been called with absolute path
                    called_path = mock_dotenv_values.call_args[0][0]
                    assert called_path.is_absolute()
                    assert called_path.parent.name == "relative"
