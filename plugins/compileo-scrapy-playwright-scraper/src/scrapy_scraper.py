import json
import logging
import os
import sys
from datetime import datetime, timezone

from typing import Dict, Any, List, Union, Optional
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy_playwright.page import PageMethod
import crochet
from twisted.internet import defer
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import click

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Router
plugin_router = APIRouter()

class ScrapeRequest(BaseModel):
    url: str
    depth: int = 1
    config: Dict[str, Any] = {}
    project_id: str = "default"

@plugin_router.post("/scrape")
async def api_scrape(request: ScrapeRequest):
    """Scrape a website."""
    logger.info(f"DEBUG: api_scrape received request: {request}")
    handler = ScrapyPlaywrightHandler()
    if not handler.can_handle(request.url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        # Create document record in DB first
        doc_id = None
        try:
            # Add src directory to path for plugin import context
            import os
            plugin_dir = os.path.dirname(__file__)  # plugins/.../src/
            src_dir = os.path.abspath(os.path.join(plugin_dir, "../../../src"))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

            # Try to import dynamically to avoid plugin loading issues
            import importlib
            try:
                db_module = importlib.import_module('compileo.storage.src.database')
                repo_module = importlib.import_module('compileo.storage.src.project.database_repositories')
                db = db_module.get_db_connection()
                repo = repo_module.DocumentRepository(db)
                doc_id = repo.create_document(request.project_id, request.url)
                logger.info(f"Created document record {doc_id} for {request.url}")
            except ImportError as ie:
                logger.warning(f"Dynamic import failed: {ie}, trying direct import")
                # Fallback: try direct import
                from compileo.storage.src.database import get_db_connection
                from compileo.storage.src.project.database_repositories import DocumentRepository
                db = get_db_connection()
                repo = DocumentRepository(db)
                doc_id = repo.create_document(int(request.project_id), request.url)
                logger.info(f"Created document record {doc_id} for {request.url} (fallback import)")
        except Exception as e:
            logger.error(f"Failed to create document record in DB: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create document record: {str(e)}")

        # Run scraping in a subprocess to avoid Twisted reactor conflicts with Uvicorn/FastAPI
        import subprocess

        # Use virtual environment python if available
        python_exe = sys.executable
        # Check for common virtual environment patterns in parent directories
        current_dir = os.getcwd()
        check_dirs = [current_dir]
        parent = os.path.dirname(current_dir)
        while parent and parent != current_dir:
            check_dirs.append(parent)
            current_dir = parent
            parent = os.path.dirname(current_dir)

        for check_dir in check_dirs:
            venv_paths = [
                os.path.join(check_dir, '.venv', 'bin', 'python'),
                os.path.join(check_dir, 'venv', 'bin', 'python'),
                os.path.join(check_dir, 'env', 'bin', 'python')
            ]
            for venv_python in venv_paths:
                if os.path.exists(venv_python):
                    python_exe = venv_python
                    break
            if python_exe != sys.executable:
                break

        # Prepare command with document_id
        cmd = [
            python_exe,
            os.path.abspath(__file__),
            "--url", request.url,
            "--depth", str(request.depth),
            "--project-id", str(request.project_id),
            "--document-id", str(doc_id),
            "--config-json", json.dumps(request.config)
        ]

        # Setup environment with PYTHONPATH to include project root
        env = os.environ.copy()
        # Correctly point to Compileo root (up 4 levels from src/scrapy_scraper.py)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        # We need to add 'src' to PYTHONPATH so 'compileo' can be imported directly
        src_path = os.path.join(project_root, "src")

        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_path}:{project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = f"{src_path}:{project_root}"

        logger.info(f"Running scraper subprocess: {' '.join(cmd)}")

        # Run synchronously (blocking the request, but safe for reactor)
        # In production, this should be a background task or job queue
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        # Log output for debugging
        if result.stdout:
            logger.info(f"Scraper stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Scraper stderr: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"Scraper subprocess failed: {result.stderr}")
            raise Exception(f"Scraper failed: {result.stderr}")

        return {"status": "success", "message": "Scraping completed", "url": request.url}
    except Exception as e:
        logger.error(f"Scraping API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@plugin_router.get("/config")
async def get_config():
    """Get configuration options."""
    return {
        "depth_options": [1, 2, 3, 4, 5],
        "wait_options": ["load", "domcontentloaded", "networkidle"],
        "default_config": {
            "wait_for_load": "networkidle",
            "scroll_to_bottom": False
        }
    }

# CLI Command
@click.command("scrape-website")
@click.option('--url', '-u', required=True, help='Website URL to scrape')
@click.option('--depth', '-d', type=int, default=1, help='Scraping depth (1-5)')
@click.option('--project-id', '-p', required=True, help='Project ID')
@click.option('--document-id', help='Existing document ID (if not provided, will create new)', default=None)
@click.option('--wait-for', default='networkidle',
              type=click.Choice(['load', 'domcontentloaded', 'networkidle']))
@click.option('--scroll-to-bottom', is_flag=True, help='Scroll to load dynamic content')
@click.option('--config-json', help='JSON configuration string (overrides other flags)', default=None)
def scrape_website_cli(url, depth, project_id, document_id, wait_for, scroll_to_bottom, config_json):
    """Scrape website and process through ingestion pipeline."""
    handler = ScrapyPlaywrightHandler()
    if not handler.can_handle(url):
        click.echo("Error: Invalid URL", err=True)
        return

    if config_json:
        try:
            config = json.loads(config_json)
            # Ensure depth is set in config if passed as arg
            if 'depth_limit' not in config:
                config['depth_limit'] = depth
        except json.JSONDecodeError:
            click.echo("Error: Invalid JSON config", err=True)
            return
    else:
        config = {
            'depth_limit': depth,
            'wait_for_load': wait_for,
            'scroll_to_bottom': scroll_to_bottom
        }
    
    click.echo(f"Scraping {url}...")
    try:
        # Use provided document_id or create new document record in DB
        doc_id = document_id
        if doc_id is None:
            try:
                from compileo.storage.src.database import get_db_connection
                from compileo.storage.src.project.database_repositories import DocumentRepository
                db = get_db_connection()
                repo = DocumentRepository(db)
                doc_id = repo.create_document(project_id, url)
                logger.info(f"Created document record {doc_id} for {url}")
            except Exception as e:
                logger.error(f"Failed to create document record in DB: {e}")
                # Fallback to timestamp ID if DB creation fails
                doc_id = int(datetime.now(timezone.utc).timestamp())
        else:
            logger.info(f"Using provided document ID {doc_id} for {url}")

        content = handler.process(url, config=config, document_id=doc_id, project_id=project_id)
        if content:
            click.echo(f"Successfully scraped {len(content)} characters.")
        else:
            click.echo("Scraping failed or returned no content.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

class ScrapyPlaywrightHandler:
    """
    Plugin handler for website ingestion.
    """
    
    def can_handle(self, input_path: str) -> bool:
        """Check if this handler can process the input."""
        return input_path.startswith(('http://', 'https://'))

    def process(self, url: str, **kwargs) -> Optional[str]:
        """
        Process the URL and return the content.
        Compatible with ingestion.main.parse_document signature.
        """
        config = kwargs.get('config', {})
        # Map flat kwargs to config if needed (e.g. depth from CLI)
        if 'depth' in kwargs:
            config['depth_limit'] = kwargs['depth']
            
        scraper = ScrapyPlaywrightScraper()
        try:
            result = scraper.scrape(url, config)
        except Exception as e:
            raise e
        
        if url not in result:
            logger.error(f"No result returned for URL {url}. Result keys: {list(result.keys())}")
            return None
            
        data = result[url]
        if "error" in data:
            logger.error(f"Scraping error in data for {url}: {data['error']}")
            
        content = data.get('main_content', '')
        if not content:
             logger.warning(f"Empty content for {url}. Metadata: {data.get('metadata')}")
        
        # Save results if we have a document_id
        document_id = kwargs.get('document_id')
        if document_id is not None:
            project_id = kwargs.get('project_id', 'unknown')
            self._save_results(document_id, project_id, content, data)
            
        return content

    def _save_results(self, document_id, project_id, content, data):
        # Create parsed directory
        parsed_dir = f"storage/parsed/{project_id}"
        os.makedirs(parsed_dir, exist_ok=True)
        
        # Use timestamp for filenames to match standard file ingestion naming convention
        timestamp_name = int(datetime.now(timezone.utc).timestamp())
        parsed_filename = f"{timestamp_name}_1.json"
        parsed_file_path = os.path.join(parsed_dir, parsed_filename)
        
        # Create JSON structure
        json_structure = {
            "content_type": data.get("content_type", "main_content_only"),
            "main_content": content,
            "metadata": {
                "chunk_number": 1,
                "document_id": str(document_id),
                "processing_stage": "parsing",
                **data.get("metadata", {})
            }
        }
        
        # Save parsed content
        with open(parsed_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved parsed JSON for website: {parsed_file_path}")
        
        # Create manifest
        manifest_data = {
            "original_file": data.get("metadata", {}).get("source_url", ""),
            "file_type": "url",
            "file_size": 0,
            "total_pages": 1,
            "parser_used": "scrapy_playwright",
            "parsing_type": "website",
            "splitting_occurred": False,
            "document_id": document_id,
            "project_id": project_id,
            "content_length": len(content),
            "created_at": str(datetime.now(timezone.utc)),
            "parsed_files": [parsed_file_path],
            "splits": [
                {
                    "chunk_number": 1,
                    "file_path": parsed_file_path,
                    "content_type": "website",
                    "overlap": {
                        "with_previous": None,
                        "with_next": None
                    }
                }
            ]
        }
        
        manifest_filename = f"{timestamp_name}_manifest.json"
        manifest_path = os.path.join(parsed_dir, manifest_filename)
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
        # Update DB - we need to import this from core logic or duplicate it?
        # Ideally we should use the core function if available or implement DB update here.
        # Since this is a plugin, importing from core src is allowed if in path.
        try:
            from compileo.features.ingestion.main import _update_document_with_single_file_manifest
            _update_document_with_single_file_manifest(document_id, manifest_data)
        except ImportError as e:
            logger.warning(f"Could not update document database record - import failed: {e}")
            import sys
            logger.warning(f"sys.path: {sys.path}")


class ScrapyPlaywrightScraper:
    """
    Website scraper using scrapy-playwright for JavaScript-enabled content extraction.
    """

    def __init__(self):
        self.settings = self._get_scrapy_settings()

    def _get_scrapy_settings(self) -> Dict[str, Any]:
        """Configure scrapy-playwright settings."""
        return {
            'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
            'DOWNLOAD_HANDLERS': {
                'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
                'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            },
            'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
            'PLAYWRIGHT_LAUNCH_OPTIONS': {
                'headless': True,
                'args': ['--no-sandbox', '--disable-dev-shm-usage'],
            },
            'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 3000000,  # 50 minutes
            'ROBOTSTXT_OBEY': True,
            'USER_AGENT': 'Compileo-Web-Scraper/1.0',
            'LOG_LEVEL': 'ERROR', # Reduce noise
        }

    @crochet.wait_for(timeout=300.0) # 5 minutes timeout
    def scrape(self, urls: Union[str, List[str]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Scrape website(s) and return content in parsing-compatible format.
        """
        if isinstance(urls, str):
            urls = [urls]

        config = config or {}
        
        from scrapy.crawler import CrawlerRunner
        
        runner = CrawlerRunner(self.settings)
        deferreds = []
        
        for url in urls:
            result_container = {}
            d = runner.crawl(WebsiteScraperSpider, url=url, config=config, result_container=result_container)
            deferreds.append((d, result_container, url))

        d_list = [d[0] for d in deferreds]
        
        def collect_results(_):
            final_results = {}
            for _, container, url in deferreds:
                if url in container:
                    final_results[url] = container[url]
                else:
                    final_results[url] = {"error": "Scraping failed or produced no content"}
            return final_results

        return defer.DeferredList(d_list).addCallback(collect_results)

class WebsiteScraperSpider(scrapy.Spider):
    """Scrapy spider for website content extraction."""

    name = 'website_scraper'

    def __init__(self, url=None, config=None, result_container=None, *args, **kwargs):
        super(WebsiteScraperSpider, self).__init__(*args, **kwargs)
        self.start_url = url
        self.config = config or {}
        self.result_container = result_container if result_container is not None else {}

    def start_requests(self):
        """Start scraping with playwright enabled."""
        print(f"DEBUG: Spider starting for URL: {self.start_url}")
        if not self.start_url:
            return

        meta = {
            'playwright': True,
            'playwright_include_page': True,
            'playwright_page_methods': [
                PageMethod('wait_for_load_state', state=self.config.get('wait_for_load', 'networkidle')),
            ] + self._get_additional_methods(),
        }
        
        yield scrapy.Request(
            url=self.start_url,
            callback=self.parse,
            meta=meta,
            errback=self.errback,
        )

    def _get_additional_methods(self) -> List[PageMethod]:
        """Get additional page methods based on config."""
        methods = []

        # Wait for specific selectors if configured
        if 'wait_selector' in self.config and self.config['wait_selector']:
            methods.append(PageMethod('wait_for_selector', self.config['wait_selector']))

        # Scroll to load dynamic content
        if self.config.get('scroll_to_bottom', False):
            methods.append(PageMethod('evaluate', 'window.scrollTo(0, document.body.scrollHeight)'))
            methods.append(PageMethod('wait_for_timeout', 1000))

        return methods

    async def parse(self, response):
        """Extract content from scraped page."""
        print(f"DEBUG: Parse method started for {response.url}")
        page = response.meta["playwright_page"]
        
        try:
            timestamp = int(datetime.now(timezone.utc).timestamp())
            
            # Extract main content
            main_content = await self._extract_main_content(page)
            title = await page.title()

            # Create metadata
            metadata = self._create_metadata(response.url, title)

            # Format as parsing-compatible output
            result = {
                'content_type': 'main_content_only',
                'main_content': main_content,
                'metadata': metadata
            }

            self.result_container[self.start_url] = result
        finally:
            await page.close()

    async def errback(self, failure):
        print(f"DEBUG: Errback called for {failure.request.url}: {failure.value}")
        page = failure.request.meta.get("playwright_page")
        if page:
            await page.close()
        logger.error(f"Scraping failed for {failure.request.url}: {failure.value}")

    async def _extract_main_content(self, page) -> str:
        """Extract and clean main content from HTML using Playwright."""
        # Remove unwanted elements
        await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('script, style, nav, header, footer, aside, iframe, noscript');
                elements.forEach(el => el.remove());
            }
        """)

        # Extract text using innerText
        content = await page.evaluate("""
            () => {
                const selectors = ['main', 'article', '[role="main"]', '.content', '.main-content', '#content', 'body'];
                for (const selector of selectors) {
                    const element = document.querySelector(selector);
                    if (element && element.innerText.trim().length > 0) {
                        return element.innerText;
                    }
                }
                return document.body.innerText;
            }
        """)

        return self._clean_content(content)

    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        if not content:
            return ""
        
        lines = [line.strip() for line in content.splitlines()]
        content = '\n'.join([line for line in lines if line])
        
        return content

    def _create_metadata(self, url: str, title: str) -> Dict[str, Any]:
        """Create metadata for the scraped content."""
        return {
            'source_url': url,
            'scraped_at': datetime.now(timezone.utc).isoformat(),
            'title': title or '',
            'content_type': 'website',
            'processing_stage': 'scraping',
            'page_range': {
                'start': 1,
                'end': 1,
                'actual_pages': 1
            }
        }
if __name__ == "__main__":
    # Install AsyncioSelectorReactor BEFORE importing crochet or other twisted modules
    # This is strictly required for scrapy-playwright to work
    if "twisted.internet.reactor" not in sys.modules:
        try:
            from twisted.internet import asyncioreactor
            asyncioreactor.install()
        except Exception as e:
            # If it fails (e.g. already installed), we log and proceed
            print(f"Warning: Could not install AsyncioSelectorReactor: {e}")
            
    # Initialize crochet to run twisted reactor in a separate thread/loop
    crochet.setup()
    
    scrape_website_cli()