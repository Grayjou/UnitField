#!/usr/bin/env python3
"""
Unified Pure Python Publisher for PyPI

A transparent, secure, and cross-platform pure Python solution for building
and uploading Python packages to PyPI/TestPyPI. This script provides a single-file
alternative to the PowerShell-based scripts, with support for TOML configuration.

Features:
- Pure Python (no PowerShell or shell script dependencies)
- Cross-platform (works on Windows, macOS, Linux)
- Single-file TOML configuration support
- Secure token handling (environment variables, keyring, or config file)
- Transparent build and upload process with detailed logging
- Support for both TestPyPI and PyPI
- Clean build artifacts before building
- Version detection and validation
- Dry-run mode for testing

Usage:
    python unified_publish.py                    # Interactive mode
    python unified_publish.py --config my.toml   # Use config file
    python unified_publish.py --test             # Publish to TestPyPI
    python unified_publish.py --prod             # Publish to PyPI
    python unified_publish.py --dry-run          # Test without uploading
    python unified_publish.py --no-confirm       # Skip confirmation prompts

Configuration (publish_config.toml):
    [publish]
    target = "test"  # "test" or "prod"
    confirm = true   # Whether to require confirmation
    clean = true     # Clean dist/ before building
    
    [package]
    path = "."       # Path to package root
    exclude = ["tests", "docs"]  # Patterns to exclude
    
    [tokens]
    # Tokens can be specified here (not recommended for security)
    # Better: use environment variables PYPI_TOKEN / TESTPYPI_TOKEN
    # Or: use system keyring via kx-publish-pypi setup-tokens

Author: KhaderX.com
License: MIT
"""

__version__ = "1.0.0"

import os
import sys
import re
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

# Try to import optional dependencies with fallbacks
# tomllib is available in Python 3.11+, tomli provides compatibility for earlier versions
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore # Fallback for Python < 3.11
    except ImportError:
        tomllib = None  # TOML support disabled

try:
    import keyring
except ImportError:
    keyring = None


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.CYAN = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_step(emoji: str, message: str, color: str = Colors.CYAN) -> None:
    """Print a step message with emoji and color."""
    print(f"{color}{emoji} {message}{Colors.RESET}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")


def print_banner() -> None:
    """Print a welcome banner."""
    width = 60
    print()
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{'🚀 Unified Pure Python Publisher 🚀':^{width}}{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print(f"{'Transparent • Secure • Cross-Platform':^{width}}")
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print()


@dataclass
class PublishConfig:
    """Configuration for the publish process."""
    # Target environment
    target: str = "test"  # "test" or "prod"
    confirm: bool = True
    clean: bool = True
    dry_run: bool = False
    
    # Package settings
    package_path: Path = field(default_factory=lambda: Path("."))
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Tokens (loaded from various sources)
    test_token: Optional[str] = None
    prod_token: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.package_path, str):
            self.package_path = Path(self.package_path)


def load_config_from_toml(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Dictionary with configuration values
    """
    if tomllib is None:
        print_warning("tomllib/tomli not available. Install with: pip install tomli")
        return {}
    
    if not config_path.exists():
        return {}
    
    try:
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return {}


def get_token_from_keyring(env: str) -> Optional[str]:
    """Get token from system keyring.
    
    Args:
        env: Environment name ("testpypi" or "pypi")
        
    Returns:
        Token string if found, None otherwise
    """
    if keyring is None:
        return None
    
    service_name = f"kx-publish-{env}"
    try:
        return keyring.get_password(service_name, "__token__")
    except Exception:
        return None


def get_token(env: str, config: PublishConfig) -> Optional[str]:
    """Get API token from various sources in order of preference.
    
    Priority:
    1. Environment variable (PYPI_TOKEN / TESTPYPI_TOKEN)
    2. Config object (if loaded from TOML)
    3. System keyring (via kx-publish-pypi)
    
    Args:
        env: Environment name ("test" or "prod")
        config: PublishConfig object
        
    Returns:
        Token string if found, None otherwise
    """
    # Map short names to full names
    env_map = {"test": "testpypi", "prod": "pypi"}
    full_env = env_map.get(env, env)
    
    # 1. Check environment variables
    env_var = f"{full_env.upper()}_TOKEN"
    token = os.environ.get(env_var)
    if token:
        print_info(f"Using token from environment variable: {env_var}")
        return token
    
    # 2. Check config object
    if env == "test" and config.test_token:
        print_info("Using token from configuration")
        return config.test_token
    elif env == "prod" and config.prod_token:
        print_info("Using token from configuration")
        return config.prod_token
    
    # 3. Check system keyring
    token = get_token_from_keyring(full_env)
    if token:
        print_info(f"Using token from system keyring")
        return token
    
    return None


def read_pyproject_toml(package_path: Path) -> Optional[Dict[str, Any]]:
    """Read pyproject.toml from the package directory.
    
    Args:
        package_path: Path to the package root
        
    Returns:
        Dictionary with pyproject.toml content, or None on error
    """
    pyproject_path = package_path / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    
    if tomllib is None:
        # Fallback to regex parsing for version only
        try:
            content = pyproject_path.read_text(encoding="utf-8")
            return {"_raw": content}
        except Exception:
            return None
    
    try:
        with pyproject_path.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print_error(f"Failed to read pyproject.toml: {e}")
        return None


def detect_version(package_path: Path) -> Tuple[Optional[str], str]:
    """Detect package version from various sources.
    
    Args:
        package_path: Path to the package root
        
    Returns:
        Tuple of (version, detection_method)
    """
    data = read_pyproject_toml(package_path)
    if data is None:
        return None, "pyproject.toml not found"
    
    # Try static version
    project = data.get("project", {})
    if "version" in project:
        return project["version"], "static in pyproject.toml"
    
    # Try dynamic version detection
    dynamic = project.get("dynamic", [])
    if "version" in dynamic:
        # Check setuptools dynamic configuration
        tool = data.get("tool", {})
        setuptools = tool.get("setuptools", {})
        dynamic_cfg = setuptools.get("dynamic", {})
        version_cfg = dynamic_cfg.get("version", {})
        
        # Try file-based version
        version_file = version_cfg.get("file")
        if version_file:
            version_path = package_path / version_file
            version = read_version_from_file(version_path)
            if version:
                return version, f"file: {version_file}"
        
        # Try attribute-based version
        version_attr = version_cfg.get("attr")
        if version_attr:
            version = resolve_attribute_version(package_path, version_attr)
            if version:
                return version, f"attr: {version_attr}"
    
    # Fallback: search for __version__.py files
    patterns = [
        "src/*/__version__.py",
        "*/__version__.py",
        "__version__.py",
    ]
    for pattern in patterns:
        for version_file in package_path.glob(pattern):
            version = read_version_from_file(version_file)
            if version:
                return version, f"file: {version_file.relative_to(package_path)}"
    
    return None, "could not detect version"


def read_version_from_file(file_path: Path) -> Optional[str]:
    """Read __version__ from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Version string if found, None otherwise
    """
    if not file_path.exists():
        return None
    
    try:
        content = file_path.read_text(encoding="utf-8")
        match = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
            content,
            re.MULTILINE
        )
        return match.group(1) if match else None
    except Exception:
        return None


def resolve_attribute_version(package_path: Path, attr_path: str) -> Optional[str]:
    """Resolve version from an attribute path.
    
    Args:
        package_path: Path to the package root
        attr_path: Attribute path like "mypackage.__version__" or "mypackage.version"
                   Can also be in format "module:attribute" which will be normalized.
        
    Returns:
        Version string if found, None otherwise
    """
    # Handle pyproject.toml format: package.__version__:__version__ or package:__version__
    # The colon separates module path from attribute name, normalize to dots for path resolution
    attr_path = attr_path.replace(":", ".")
    parts = attr_path.split(".")
    
    # Try to find the module file
    for base in [package_path / "src", package_path]:
        module_path = base
        for part in parts[:-1]:
            module_path = module_path / part
        
        # Try as a .py file
        py_file = module_path.with_suffix(".py")
        if py_file.exists():
            version = read_version_from_file(py_file)
            if version:
                return version
        
        # Try as __init__.py in a directory
        init_file = module_path / "__init__.py"
        if init_file.exists():
            version = read_version_from_file(init_file)
            if version:
                return version
    
    return None


def clean_build_artifacts(package_path: Path) -> None:
    """Clean build artifacts (dist/, build/, *.egg-info).
    
    Args:
        package_path: Path to the package root
    """
    print_step("🧹", "Cleaning build artifacts...")
    
    dirs_to_clean = ["dist", "build"]
    patterns_to_clean = ["*.egg-info"]
    
    cleaned = []
    
    for dir_name in dirs_to_clean:
        dir_path = package_path / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            cleaned.append(dir_name)
    
    for pattern in patterns_to_clean:
        for path in package_path.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                cleaned.append(path.name)
    
    if cleaned:
        print_success(f"Cleaned: {', '.join(cleaned)}")
    else:
        print_info("No artifacts to clean")


def build_package(package_path: Path) -> bool:
    """Build the package using python -m build.
    
    Args:
        package_path: Path to the package root
        
    Returns:
        True if build succeeded, False otherwise
    """
    print_step("🏗️", "Building package...")
    
    # Check if build module is available
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--version"],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("The 'build' package is not installed.")
        print_info("Install with: pip install build")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "build"],
            cwd=str(package_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_error("Build failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
        
        print_success("Package built successfully!")
        
        # List built artifacts
        dist_path = package_path / "dist"
        if dist_path.exists():
            artifacts = list(dist_path.glob("*"))
            for artifact in artifacts:
                print_info(f"  Built: {artifact.name}")
        
        return True
        
    except Exception as e:
        print_error(f"Build failed with error: {e}")
        return False


def upload_package(
    package_path: Path,
    token: str,
    target: str,
    dry_run: bool = False
) -> Tuple[bool, Optional[str]]:
    """Upload package to PyPI or TestPyPI.
    
    Args:
        package_path: Path to the package root
        token: API token for authentication
        target: Target repository ("test" or "prod")
        dry_run: If True, don't actually upload
        
    Returns:
        Tuple of (success: bool, package_name: Optional[str])
    """
    repo_urls = {
        "test": "https://test.pypi.org/legacy/",
        "prod": "https://upload.pypi.org/legacy/",
    }
    repo_names = {
        "test": "TestPyPI",
        "prod": "PyPI",
    }
    
    repo_url = repo_urls.get(target)
    repo_name = repo_names.get(target, target)
    
    if not repo_url:
        print_error(f"Unknown target: {target}")
        return False, None
    
    dist_path = package_path / "dist"
    if not dist_path.exists():
        print_error("dist/ directory not found. Run build first.")
        return False, None
    
    artifacts = list(dist_path.glob("*.whl")) + list(dist_path.glob("*.tar.gz"))
    if not artifacts:
        print_error("No package files found in dist/")
        return False, None
    
    # Extract package name from the first artifact using regex
    # Package names can contain hyphens, so we need to find the version pattern
    # Wheels: package_name-version-py3-...-any.whl
    # Tar.gz: package_name-version.tar.gz
    package_name = None
    if artifacts:
        artifact = artifacts[0]
        name = artifact.stem
        # Remove .tar suffix if present (for .tar.gz files)
        if name.endswith(".tar"):
            name = name[:-4]
        # Use regex to find version pattern
        # PEP 440 versions can start with digits and include pre-release tags
        # Examples: 1.0, 1.0.0, 1.0.0a1, 1.0.0rc1, 1.0.0.post1, 1.0.0.dev1
        match = re.match(r'^(.+?)-(\d+(?:\.\d+)*(?:[a-zA-Z]+\d*)?(?:\.\w+)*)(?:-|$)', name)
        if match:
            package_name = match.group(1)
        else:
            # Fallback: use first part before hyphen
            package_name = name.split("-")[0]
    
    if dry_run:
        print_step("🔍", f"DRY RUN: Would upload to {repo_name}")
        for artifact in artifacts:
            print_info(f"  Would upload: {artifact.name}")
        return True, package_name
    
    print_step("📤", f"Uploading to {repo_name}...")
    
    # Check if twine is available
    try:
        subprocess.run(
            [sys.executable, "-m", "twine", "--version"],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("The 'twine' package is not installed.")
        print_info("Install with: pip install twine")
        return False, None
    
    try:
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository-url", repo_url,
            "--username", "__token__",
            "--password", token,
        ]
        cmd.extend([str(f) for f in artifacts])
        
        result = subprocess.run(
            cmd,
            cwd=str(package_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_error("Upload failed!")
            if result.stderr:
                # Check for common errors
                stderr_lower = result.stderr.lower()
                if "already exists" in stderr_lower or "400" in stderr_lower:
                    print_warning("Version already exists on the repository.")
                    print_info("Bump version and rebuild: python -m build")
                elif "403" in stderr_lower or "forbidden" in stderr_lower:
                    print_warning("Authentication failed. Check your API token.")
                else:
                    print(f"Error: {result.stderr}")
            return False, None
        
        print_success(f"Package uploaded to {repo_name}!")
        
        # Print package URL with actual package name
        if package_name:
            if target == "test":
                print_info(f"View at: https://test.pypi.org/project/{package_name}/")
            else:
                print_info(f"View at: https://pypi.org/project/{package_name}/")
        
        return True, package_name
        
    except Exception as e:
        print_error(f"Upload failed with error: {e}")
        return False, None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Pure Python Publisher for PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python unified_publish.py                    # Interactive mode
    python unified_publish.py --test             # Publish to TestPyPI
    python unified_publish.py --prod             # Publish to PyPI
    python unified_publish.py --dry-run --prod   # Test without uploading
    python unified_publish.py --config my.toml   # Use config file
    python unified_publish.py --no-confirm       # Skip confirmation
        """,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("publish_config.toml"),
        help="Path to TOML configuration file (default: publish_config.toml)",
    )
    
    parser.add_argument(
        "--path", "-p",
        type=Path,
        default=Path("."),
        help="Path to package root (default: current directory)",
    )
    
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--test", "-t",
        action="store_true",
        help="Publish to TestPyPI",
    )
    target_group.add_argument(
        "--prod",
        action="store_true",
        help="Publish to production PyPI",
    )
    
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts",
    )
    
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean build artifacts before building",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build but don't upload (test mode)",
    )
    
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building (use existing dist/)",
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> PublishConfig:
    """Load configuration from file and command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        PublishConfig object
    """
    config = PublishConfig()
    
    # Load from TOML file if it exists
    if args.config.exists():
        print_info(f"Loading config from: {args.config}")
        toml_data = load_config_from_toml(args.config)
        
        # Apply TOML settings
        publish_cfg = toml_data.get("publish", {})
        config.target = publish_cfg.get("target", config.target)
        config.confirm = publish_cfg.get("confirm", config.confirm)
        config.clean = publish_cfg.get("clean", config.clean)
        
        package_cfg = toml_data.get("package", {})
        if "path" in package_cfg:
            config.package_path = Path(package_cfg["path"])
        config.exclude_patterns = package_cfg.get("exclude", [])
        
        tokens_cfg = toml_data.get("tokens", {})
        config.test_token = tokens_cfg.get("test")
        config.prod_token = tokens_cfg.get("prod")
    
    # Override with command-line arguments
    if args.path:
        config.package_path = args.path
    
    if args.test:
        config.target = "test"
    elif args.prod:
        config.target = "prod"
    
    if args.no_confirm:
        config.confirm = False
    
    if args.no_clean:
        config.clean = False
    
    if args.dry_run:
        config.dry_run = True
    
    return config


def validate_package(package_path: Path) -> bool:
    """Validate that the package has required files.
    
    Args:
        package_path: Path to the package root
        
    Returns:
        True if validation passed, False otherwise
    """
    print_step("🔍", "Validating package...")
    
    required_files = ["pyproject.toml"]
    recommended_files = ["README.md", "LICENSE"]
    
    all_ok = True
    
    for file_name in required_files:
        file_path = package_path / file_name
        if file_path.exists():
            print_success(f"Found: {file_name}")
        else:
            print_error(f"Missing required: {file_name}")
            all_ok = False
    
    for file_name in recommended_files:
        file_path = package_path / file_name
        if file_path.exists():
            print_success(f"Found: {file_name}")
        else:
            print_warning(f"Missing recommended: {file_name}")
    
    return all_ok


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()
    
    if args.version:
        print(f"Unified Pure Python Publisher v{__version__}")
        return 0
    
    print_banner()
    
    # Load configuration
    config = load_config(args)
    package_path = config.package_path.resolve()
    
    print_info(f"Package path: {package_path}")
    print_info(f"Target: {config.target.upper()}")
    if config.dry_run:
        print_warning("DRY RUN MODE - No actual upload will occur")
    print()
    
    # Validate package
    if not validate_package(package_path):
        print_error("Package validation failed!")
        return 1
    
    # Detect version
    version, method = detect_version(package_path)
    if version:
        print_success(f"Version: {version} (detected via {method})")
    else:
        print_warning(f"Could not detect version: {method}")
    print()
    
    # Confirmation
    target_name = "TestPyPI" if config.target == "test" else "PyPI"
    if config.confirm and not config.dry_run:
        print(f"\n{Colors.BOLD}Ready to build and publish to {target_name}{Colors.RESET}")
        if version:
            print(f"Version: {version}")
        
        response = input("\nProceed? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print_warning("Aborted by user.")
            return 0
        print()
    
    # Get API token
    if not config.dry_run:
        token = get_token(config.target, config)
        if not token:
            print_error(f"No API token found for {target_name}!")
            env_var = "TESTPYPI_TOKEN" if config.target == "test" else "PYPI_TOKEN"
            print_info("Set token via:")
            print_info(f"  1. Environment variable: {env_var}")
            print_info("  2. System keyring: kx-publish-pypi setup-tokens")
            print_info("  3. Config file: [tokens] section in TOML")
            return 1
    else:
        token = "DRY_RUN_TOKEN"
    
    # Clean build artifacts
    if config.clean and not args.skip_build:
        clean_build_artifacts(package_path)
        print()
    
    # Build package
    if not args.skip_build:
        if not build_package(package_path):
            return 1
        print()
    else:
        print_info("Skipping build (--skip-build)")
        print()
    
    # Upload package
    success, package_name = upload_package(package_path, token, config.target, config.dry_run)
    if not success:
        return 1
    
    print()
    if config.dry_run:
        print_success("Dry run completed successfully!")
    else:
        print_success(f"Successfully published to {target_name}!")
        if config.target == "test" and package_name:
            print_info(f"Test your package: pip install -i https://test.pypi.org/simple/ {package_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())