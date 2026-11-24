import logging
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from src.compileo.features.plugin.manager import plugin_manager
from ..core.dependencies import get_current_active_user, get_current_admin_user
from ..models.user import User as UserSchema

router = APIRouter()
logger = logging.getLogger(__name__)

# Dynamically load plugin routers
try:
    plugin_routers = plugin_manager.get_extensions("compileo.api.router")
    for plugin_id, router_obj in plugin_routers.items():
        router.include_router(router_obj, prefix=f"/{plugin_id}", tags=[plugin_id])
except Exception as e:
    # Logger might not be fully configured yet if at module level, but print/pass
    logger.error(f"Error loading plugin routes: {e}")

class PluginInfo(BaseModel):
    id: str
    name: str
    version: str
    author: str
    description: str
    entry_point: str
    extensions: Dict[str, Dict[str, str]]
    owner_id: Optional[str] = None

@router.get("/", response_model=List[PluginInfo])
async def list_plugins(current_user: UserSchema = Depends(get_current_active_user)):
    """
    List all installed plugins.
    """
    logger.debug(f"DEBUG: list_plugins called by user {current_user.id} (admin: {current_user.is_admin})")
    try:
        # For non-admins, filter by owner_id. For admins, show all.
        owner_id = current_user.id if not current_user.is_admin else None
        plugins = plugin_manager.get_all_plugins(owner_id=owner_id)
        logger.debug(f"DEBUG: Found {len(plugins)} plugins")
        return plugins
    except Exception as e:
        logger.debug(f"DEBUG: Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_plugin(
    file: UploadFile = File(...),
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Upload and install a plugin from a zip file.
    """
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    # Save uploaded file to temp
    temp_zip_path = Path(f"storage/temp_{file.filename}")
    try:
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Install plugin
        plugin_id = plugin_manager.install_plugin(str(temp_zip_path), owner_id=current_user.id)
        
        return {"status": "success", "message": f"Plugin {plugin_id} installed successfully", "plugin_id": plugin_id}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp zip
        if temp_zip_path.exists():
            os.remove(temp_zip_path)

@router.get("/dataset-formats")
async def get_dataset_formats(current_user: UserSchema = Depends(get_current_active_user)):
    """
    Get all available dataset output formats, including built-in and plugin formats.
    """
    try:
        # Built-in formats
        formats = ["jsonl", "parquet", "json"]

        # Add plugin formats
        owner_id = current_user.id if not current_user.is_admin else None
        plugin_formats = plugin_manager.get_extensions("compileo.datasetgen.formatter", owner_id=owner_id)
        formats.extend(plugin_formats.keys())

        return {"formats": formats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{plugin_id}")
async def uninstall_plugin(
    plugin_id: str,
    current_user: UserSchema = Depends(get_current_active_user)
):
    """
    Uninstall a plugin by ID.
    """
    try:
        # Check ownership
        plugin = plugin_manager.get_plugin(plugin_id)
        if plugin and plugin.get('owner_id') != current_user.id and not current_user.is_admin:
             raise HTTPException(status_code=403, detail="Access denied")

        success = plugin_manager.uninstall_plugin(plugin_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"status": "success", "message": f"Plugin {plugin_id} uninstalled"}
    except ValueError as e:
         raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
