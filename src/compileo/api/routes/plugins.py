import shutil
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.compileo.features.plugin.manager import plugin_manager

router = APIRouter()

from ..middleware.logging import logger

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

@router.get("/", response_model=List[PluginInfo])
async def list_plugins():
    """
    List all installed plugins.
    """
    try:
        return plugin_manager.get_all_plugins()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_plugin(file: UploadFile = File(...)):
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
        plugin_id = plugin_manager.install_plugin(str(temp_zip_path))
        
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
async def get_dataset_formats():
    """
    Get all available dataset output formats, including built-in and plugin formats.
    """
    try:
        # Built-in formats
        formats = ["jsonl", "parquet", "json"]

        # Add plugin formats
        plugin_formats = plugin_manager.get_extensions("compileo.datasetgen.formatter")
        formats.extend(plugin_formats.keys())

        return {"formats": formats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{plugin_id}")
async def uninstall_plugin(plugin_id: str):
    """
    Uninstall a plugin by ID.
    """
    try:
        success = plugin_manager.uninstall_plugin(plugin_id)
        if not success:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return {"status": "success", "message": f"Plugin {plugin_id} uninstalled"}
    except ValueError as e:
         raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))