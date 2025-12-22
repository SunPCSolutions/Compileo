import os
import shutil
import zipfile
import yaml
import importlib.util
import sys
import logging
import subprocess
from typing import Dict, List, Type, Any, Optional
from pydantic import BaseModel
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class PluginManifest(BaseModel):
    id: str
    name: str
    version: str
    author: str
    description: str
    entry_point: str
    extensions: Dict[str, Dict[str, str]] = {}
    format_metadata: Dict[str, Dict[str, Any]] = {}  # Optional metadata for formats
    install_script: Optional[str] = None
    uninstall_script: Optional[str] = None

class PluginManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Base directory for plugins
        self.plugins_dir = Path("./plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        # Registry: Extension Point ID -> {Key -> Class/Object}
        self.extensions: Dict[str, Dict[str, Any]] = {}

        # Loaded plugins map: Plugin ID -> Module
        self.loaded_plugins: Dict[str, Any] = {}

        # Manifests map: Plugin ID -> Manifest
        self.plugin_manifests: Dict[str, PluginManifest] = {}

        self._initialized = True
        self.load_plugins()

    def __getstate__(self):
        """Control what gets pickled - exclude non-pickleable objects."""
        state = self.__dict__.copy()
        # Don't pickle the extensions registry as it contains class objects
        # that may not be importable in other processes
        state['extensions'] = {}
        return state

    def __setstate__(self, state):
        """Restore state after unpickling and reload plugins."""
        self.__dict__.update(state)
        # Reload plugins to restore the extensions registry
        self.load_plugins()

    def install_plugin(self, zip_path: str) -> str:
        """
        Installs a plugin from a zip file.
        Returns the plugin ID upon success.
        """
        logger.info(f"Installing plugin from {zip_path}")
        
        # 1. Create a temporary extraction directory
        temp_dir = self.plugins_dir / ".temp_extract"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 2. Security Check: Zip Slip
                for member in zip_ref.namelist():
                    member_path = os.path.realpath(os.path.join(temp_dir, member))
                    if not member_path.startswith(os.path.realpath(temp_dir)):
                        raise ValueError(f"Malicious file path detected in zip: {member}")
                
                # Extract
                zip_ref.extractall(temp_dir)

            # 3. Validation: Check for plugin.yaml
            manifest_path = temp_dir / "plugin.yaml"
            if not manifest_path.exists():
                raise ValueError("Invalid plugin: missing plugin.yaml")

            # Parse Manifest
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)
            
            try:
                manifest = PluginManifest(**manifest_data)
            except Exception as e:
                raise ValueError(f"Invalid plugin.yaml format: {str(e)}")

            # 4. Atomic Install Preparation - Handle Existing Plugin
            final_dir = self.plugins_dir / manifest.id
            
            # If plugin exists, remove it first (update)
            # CRITICAL: Uninstall MUST happen BEFORE installing new dependencies.
            # Otherwise, uninstalling the old plugin might remove dependencies that the new version also needs.
            if final_dir.exists():
                logger.info(f"Updating existing plugin: {manifest.id}")
                self.uninstall_plugin(manifest.id)

            # 5. Handle Dependencies (Install for the NEW version)
            requirements_path = temp_dir / "requirements.txt"
            if requirements_path.exists():
                logger.info(f"Installing dependencies for plugin {manifest.id}")
                self._manage_dependencies(requirements_path, "install")

            # 6. Finalize Install
            shutil.move(str(temp_dir), str(final_dir))
            logger.info(f"Plugin {manifest.id} installed successfully at {final_dir}")

            # 7. Run Install Script
            if manifest.install_script:
                logger.info(f"Running install script for plugin {manifest.id}")
                try:
                    self._run_script(manifest.install_script, final_dir)
                except Exception as e:
                    logger.error(f"Install script failed for {manifest.id}: {e}")
                    # We continue as the files are already installed
            
            # Reload plugins to activate the new one
            self.load_plugins()
            
            return manifest.id

        except Exception as e:
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            logger.error(f"Plugin installation failed: {str(e)}")
            raise e

    def uninstall_plugin(self, plugin_id: str) -> bool:
        """
        Uninstalls a plugin by deleting its directory.
        """
        plugin_dir = self.plugins_dir / plugin_id
        
        # Security check: Ensure we are deleting inside plugins_dir
        if not str(plugin_dir.resolve()).startswith(str(self.plugins_dir.resolve())):
             raise ValueError("Security Violation: Attempted to delete outside plugin directory")

        if plugin_dir.exists():
            # Handle Uninstall Script (Before removing files)
            # Try to get script from loaded manifest or file
            uninstall_script = None
            if plugin_id in self.plugin_manifests:
                uninstall_script = self.plugin_manifests[plugin_id].uninstall_script
            
            if not uninstall_script:
                try:
                    manifest_path = plugin_dir / "plugin.yaml"
                    if manifest_path.exists():
                        with open(manifest_path, 'r') as f:
                            data = yaml.safe_load(f)
                            uninstall_script = data.get("uninstall_script")
                except Exception:
                    pass

            if uninstall_script:
                logger.info(f"Running uninstall script for plugin {plugin_id}")
                try:
                    self._run_script(uninstall_script, plugin_dir)
                except Exception as e:
                    logger.error(f"Uninstall script failed for {plugin_id}: {e}")

            # Handle Dependencies Removal
            requirements_path = plugin_dir / "requirements.txt"
            if requirements_path.exists():
                logger.info(f"Removing dependencies for plugin {plugin_id}")
                try:
                    self._manage_dependencies(requirements_path, "uninstall")
                except Exception as e:
                    logger.error(f"Failed to uninstall dependencies for {plugin_id}: {e}")

            # Unload from memory if loaded
            if plugin_id in self.loaded_plugins:
                del self.loaded_plugins[plugin_id]
            if plugin_id in self.plugin_manifests:
                del self.plugin_manifests[plugin_id]
            
            # Remove from extensions registry
            # Note: This is a bit expensive, might need optimization for many plugins
            # We iterate through all extensions and remove entries belonging to this plugin
            # For v1, we reload everything after uninstall which is cleaner
            
            shutil.rmtree(plugin_dir)
            logger.info(f"Plugin {plugin_id} uninstalled.")
            
            # Full reload to clean up registry
            self.load_plugins()
            return True
        return False

    def _manage_dependencies(self, requirements_path: Path, action: str):
        """
        Installs or uninstalls dependencies from a requirements file.
        Ensures dependencies are installed in the correct virtual environment.
        action: 'install' or 'uninstall'
        """
        if action not in ['install', 'uninstall']:
            raise ValueError("Invalid action. Must be 'install' or 'uninstall'")

        # Determine the correct Python executable to use
        python_exe = self._get_venv_python_executable()

        cmd = [python_exe, "-m", "pip", action, "-r", str(requirements_path)]
        if action == "uninstall":
            cmd.append("-y")
        
        # Check if we are running in a restricted environment (like Docker container as non-root)
        # and not in a virtual environment. In this case, use --user flag.
        is_venv = (os.environ.get('VIRTUAL_ENV') or
                   os.environ.get('CONDA_DEFAULT_ENV') or
                   (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)))
        
        if not is_venv and action == "install":
            # If not in venv, try to detect if we can write to site-packages
            # Simple heuristic: if we are not root, assume we might need --user
            # This is specifically for our Docker setup where we run as 'compileo' user
            import getpass
            try:
                user = getpass.getuser()
                if user != 'root':
                     # Append --user flag
                     cmd.append("--user")
                     
                     # Check for externally managed environment (PEP 668)
                     # In some Docker environments (Debian/Ubuntu based), pip install --user is still blocked
                     # unless --break-system-packages is provided.
                     try:
                         # We can check sysconfig or just detect if we are on a system that likely requires it.
                         # Since we are already isolating to --user, --break-system-packages is generally safe
                         # as we are not overwriting system files.
                         # This flag was added in pip 23.0+.
                         cmd.append("--break-system-packages")
                     except Exception:
                         pass
            except Exception:
                pass

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            logger.info(f"Successfully {action}ed dependencies from {requirements_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to {action} dependencies: {e}")
            raise e

    def _run_script(self, script: str, cwd: Path):
        """
        Runs a shell command/script in the plugin environment.
        Sets up PATH to prioritize the virtual environment.
        """
        python_exe = self._get_venv_python_executable()
        venv_bin = os.path.dirname(python_exe)
        
        # Prepare environment with venv in PATH
        env = os.environ.copy()
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        
        # Replace 'python' with specific executable if used at start
        if script.startswith("python "):
            script = script.replace("python ", f"{python_exe} ", 1)
            
        logger.info(f"Executing plugin script: {script}")
        try:
            subprocess.run(script, shell=True, check=True, cwd=cwd, env=env)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Script execution failed: {e}")

    def _get_venv_python_executable(self) -> str:
        """
        Determines the correct Python executable to use for dependency management.
        Prioritizes virtual environment Python over system Python.
        """
        # Check if we're already in a virtual environment
        venv_path = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_DEFAULT_ENV')
        if venv_path:
            venv_python = os.path.join(venv_path, 'bin', 'python')
            if os.path.exists(venv_python):
                return venv_python

        # Check for common virtual environment patterns
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            venv_paths = [
                parent / '.venv' / 'bin' / 'python',
                parent / 'venv' / 'bin' / 'python',
                parent / 'env' / 'bin' / 'python'
            ]
            for venv_python in venv_paths:
                if venv_python.exists():
                    return str(venv_python)

        # Fallback to sys.executable (current interpreter)
        logger.warning(f"Could not detect virtual environment, using current interpreter: {sys.executable}")
        return sys.executable

    def load_plugins(self):
        """
        Scans the plugins directory and loads all valid plugins.
        """
        # Reset state
        self.extensions = {}
        self.loaded_plugins = {}
        self.plugin_manifests = {}
        
        logger.info("Loading plugins...")
        
        if not self.plugins_dir.exists():
            return

        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith("."):
                try:
                    self._load_single_plugin(plugin_dir)
                except Exception as e:
                    logger.error(f"Failed to load plugin at {plugin_dir}: {str(e)}")

    def _load_single_plugin(self, plugin_dir: Path):
        manifest_path = plugin_dir / "plugin.yaml"
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)
            manifest = PluginManifest(**manifest_data)
            
            # Add plugin src to sys.path so it can import its own modules
            # We add the root of the plugin directory
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))
            
            # Dynamic Import
            entry_path_dir = plugin_dir / manifest.entry_point.replace(".", "/") / "__init__.py"
            entry_path_file = plugin_dir / f"{manifest.entry_point.replace('.', '/')}.py"
            
            spec = None
            if entry_path_dir.exists():
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{manifest.id}",
                    entry_path_dir
                )
            elif entry_path_file.exists():
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{manifest.id}",
                    entry_path_file
                )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"plugins.{manifest.id}"] = module
                spec.loader.exec_module(module)
                
                self.loaded_plugins[manifest.id] = module
                self.plugin_manifests[manifest.id] = manifest
                
                # Register Extensions
                self._register_extensions(manifest, module)
                logger.info(f"Loaded plugin: {manifest.name} ({manifest.version})")
            else:
                logger.warning(f"Could not load entry point for {manifest.id}")

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_dir.name}: {str(e)}")

    def _register_extensions(self, manifest: PluginManifest, module: Any):
        """
        Registers the extensions defined in the manifest.
        """
        for ext_point, items in manifest.extensions.items():
            if ext_point not in self.extensions:
                self.extensions[ext_point] = {}
            
            for key, class_name in items.items():
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.extensions[ext_point][key] = cls
                else:
                    logger.warning(f"Plugin {manifest.id} declares {class_name} for {ext_point}, but class not found in module.")

    def get_extensions(self, extension_point: str) -> Dict[str, Any]:
        """
        Returns a dictionary of {key: ImplementationClass} for a given extension point.
        """
        return self.extensions.get(extension_point, {})

    def get_format_metadata(self, format_type: str) -> Optional[Dict[str, Any]]:
        """
        Returns metadata for a specific format type from any installed plugin.
        """
        for manifest in self.plugin_manifests.values():
            if format_type in manifest.format_metadata:
                return manifest.format_metadata[format_type]
        return None

    def get_all_plugins(self) -> List[Dict[str, Any]]:
        """
        Returns a list of installed plugins metadata.
        """
        return [m.dict() for m in self.plugin_manifests.values()]

# Global Instance
plugin_manager = PluginManager()