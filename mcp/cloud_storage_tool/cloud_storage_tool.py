"Cloud Storage MCP tool example"

import json
import logging
import os
import sys
import jwt
import yaml
from typing import List, Dict, Any
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token, AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout, format='%(levelname)s: %(message)s')

def get_client_id() -> str:
    """
    Read the SVID JWT from file and extract the client ID from the "sub" claim.
    """
    jwt_file_path = "/opt/jwt_svid.token"
    
    content = None
    try:
        with open(jwt_file_path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        raise Exception(f"SVID JWT file {jwt_file_path} not found.")

    if content is None or content.strip() == "":
        raise Exception(f"No content in SVID JWT file {jwt_file_path}.")

    try:
        decoded = jwt.decode(content, options={"verify_signature": False})
    except jwt.DecodeError:
        raise ValueError(f"Failed to decode SVID JWT file {jwt_file_path}.")

    try:
        return decoded["sub"]
    except KeyError:
        raise KeyError('SVID JWT is missing required "sub" claim.')

# Setup GCP credentials
GCP_SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")  # JSON string or file path
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DEFAULT_BUCKET = os.getenv("DEFAULT_BUCKET")

def get_storage_client():
    """Create and return a GCS client using service account credentials."""
    try:
        if GCP_SERVICE_ACCOUNT_KEY is None:
            logger.error("GCP_SERVICE_ACCOUNT_KEY environment variable not set")
            return None
        
        # Parse service account key from JSON string or file path
        if GCP_SERVICE_ACCOUNT_KEY.startswith("{"):
            # It's a JSON string
            credentials_info = json.loads(GCP_SERVICE_ACCOUNT_KEY)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        else:
            # It's a file path
            credentials = service_account.Credentials.from_service_account_file(GCP_SERVICE_ACCOUNT_KEY)
        
        client = storage.Client(credentials=credentials, project=GCP_PROJECT_ID)
        logger.info("Successfully authenticated with GCP")
        return client
    except Exception as e:
        logger.error(f"Error authenticating with GCP: {e}")
        return None

# Create FastMCP app with auth
verifier = None
JWKS_URI = os.getenv("JWKS_URI")
ISSUER = os.getenv("ISSUER")
try:
    CLIENT_ID = get_client_id()
    if JWKS_URI is not None:
        verifier = JWTVerifier(
            jwks_uri=JWKS_URI,
            issuer=ISSUER,
            audience=CLIENT_ID
        )
except Exception as e:
    logger.warning(f"Could not set up JWT verification: {e}")

mcp = FastMCP("CloudStorage", auth=verifier)

@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_objects() -> str:
    """Get all objects from a GCS bucket."""
    logger.debug(f"Getting objects from bucket '{DEFAULT_BUCKET}'")

    # Get access token for authentication
    access_token: AccessToken | None = get_access_token()
    if access_token:
        logger.debug(f"Request authenticated with token scopes: {access_token.claims.get('scope', '')}")
    
    # Get storage client
    storage_client = get_storage_client()
    if storage_client is None:
        return json.dumps({"error": "Could not authenticate with GCP. Check GCP_SERVICE_ACCOUNT_KEY"})
    
    try:
        bucket = storage_client.bucket(DEFAULT_BUCKET)
        blobs = bucket.list_blobs()
        
        objects = []
        for blob in blobs:
            objects.append({
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "generation": blob.generation,
                "metageneration": blob.metageneration,
                "storage_class": blob.storage_class,
                "public_url": blob.public_url
            })
        
        logger.debug(f"Successfully retrieved {len(objects)} objects from bucket '{bucket_name}'")
        return json.dumps({"bucket": bucket_name, "object_count": len(objects), "objects": objects})
    
    except Exception as e:
        logger.error(f"Error listing objects from bucket '{bucket_name}': {e}")
        return json.dumps({"error": f"Failed to list objects: {str(e)}"})

@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False})
def perform_action(file_uri: str, action: str, target_uri: str) -> str:
    """
    Performs the configured action (move or copy) between cloud storage locations.
    
    Args:
        file_uri: Source GCS file path (e.g., 'path/to/file.txt')
        action: Action to perform - either 'move' or 'copy'
        target_uri: Target GCS folder path (e.g., 'folder/'). Must end with '/' to indicate it's a folder.
    """
    logger.debug(f"Performing action '{action}' from '{file_uri}' to '{target_uri}'")
    
    # Get access token for authentication
    access_token: AccessToken | None = get_access_token()
    if access_token:
        logger.debug(f"Request authenticated with token scopes: {access_token.claims.get('scope', '')}")
    
    # Validate action
    if action not in ["move", "copy"]:
        return json.dumps({"error": f"Invalid action '{action}'. Must be 'move' or 'copy'"})
    
    # Validate target is a folder (ends with /)
    if not target_uri.endswith("/"):
        return json.dumps({"error": f"Target URI must be a folder path ending with '/': {target_uri}"})
    
    # Get storage client
    storage_client = get_storage_client()
    if storage_client is None:
        return json.dumps({"error": "Could not authenticate with GCP. Check GCP_SERVICE_ACCOUNT_KEY"})
    
    try:
        if DEFAULT_BUCKET is None:
            return json.dumps({"error": "No bucket specified in file_uri and DEFAULT_BUCKET env var not set"})
        source_bucket_name = DEFAULT_BUCKET
        source_blob_name = file_uri

        target_bucket_name = DEFAULT_BUCKET
        target_folder_path = target_uri
        
        # Extract filename from source path
        filename = os.path.basename(source_blob_name)
        
        # Construct full target blob path (folder + filename)
        target_blob_name = os.path.join(target_folder_path, filename).replace("\\", "/")
        
        # Get source bucket and blob
        source_bucket = storage_client.bucket(DEFAULT_BUCKET)
        source_blob = source_bucket.blob(source_blob_name)
        
        # Check if source blob exists
        if not source_blob.exists():
            return json.dumps({"error": f"Source file does not exist: {file_uri}"})
        
        # Get target bucket
        target_bucket = storage_client.bucket(DEFAULT_BUCKET)
        
        # Construct full URIs for response
        full_source_uri = f"gs://{source_bucket_name}/{source_blob_name}"
        full_target_uri = f"gs://{target_bucket_name}/{target_blob_name}"
        
        # Perform copy operation
        target_blob = source_bucket.copy_blob(
            source_blob, target_bucket, target_blob_name
        )
        
        result = {
            "action": action,
            "source": full_source_uri,
            "target": full_target_uri,
            "target_folder": f"gs://{target_bucket_name}/{target_folder_path}",
            "filename": filename,
            "status": "success"
        }
        
        # If action is move, delete the source blob
        if action == "move":
            source_blob.delete()
            logger.debug(f"Successfully moved '{full_source_uri}' to '{full_target_uri}'")
            result["message"] = f"File moved from {full_source_uri} to {full_target_uri}"
        else:
            logger.debug(f"Successfully copied '{full_source_uri}' to '{full_target_uri}'")
            result["message"] = f"File copied from {full_source_uri} to {full_target_uri}"
        
        return json.dumps(result)
    
    except Exception as e:
        logger.error(f"Error performing {action} operation: {e}")
        return json.dumps({"error": f"Failed to {action} file: {str(e)}"})

@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def load_rules(config_uri: str) -> str:
    """Fetches and parses the YAML configuration from a cloud location or local file path."""
    logger.debug(f"Loading rules from config URI '{config_uri}'")
    
    # Get access token for authentication
    access_token: AccessToken | None = get_access_token()
    if access_token:
        logger.debug(f"Request authenticated with token scopes: {access_token.claims.get('scope', '')}")
    
    try:
        yaml_content = None
        
        # Check if it's a GCS URI
        if config_uri.startswith("gs://"):
            # Get storage client for GCS
            storage_client = get_storage_client()
            if storage_client is None:
                return json.dumps({"error": "Could not authenticate with GCP. Check GCP_SERVICE_ACCOUNT_KEY"})
            
            # Parse bucket and blob path
            uri_parts = config_uri.replace("gs://", "").split("/", 1)
            if len(uri_parts) != 2:
                return json.dumps({"error": f"Invalid GCS URI format: {config_uri}"})
            bucket_name, blob_name = uri_parts
            
            # Fetch the file from GCS
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                return json.dumps({"error": f"Config file does not exist: {config_uri}"})
            
            yaml_content = blob.download_as_text()
            logger.debug(f"Successfully downloaded config from GCS: {config_uri}")
        
        else:
            # Treat as local file path
            # Support relative paths from the script directory
            if not os.path.isabs(config_uri):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, config_uri)
            else:
                config_path = config_uri
            
            if not os.path.exists(config_path):
                return json.dumps({"error": f"Config file does not exist: {config_path}"})
            
            with open(config_path, 'r') as file:
                yaml_content = file.read()
            logger.debug(f"Successfully loaded config from local file: {config_path}")
        
        # Parse YAML content
        config = yaml.safe_load(yaml_content)
        
        # Validate the config structure
        if not isinstance(config, dict) or "rules" not in config:
            return json.dumps({"error": "Invalid config format. Expected 'rules' key at root level"})
        
        rules = config.get("rules", [])
        if not isinstance(rules, list):
            return json.dumps({"error": "Invalid config format. 'rules' must be a list"})
        
        # Validate each rule has required fields
        for i, rule in enumerate(rules):
            required_fields = ["pattern", "target", "action"]
            missing_fields = [field for field in required_fields if field not in rule]
            if missing_fields:
                return json.dumps({
                    "error": f"Rule {i} is missing required fields: {missing_fields}"
                })
            
            if rule["action"] not in ["move", "copy"]:
                return json.dumps({
                    "error": f"Rule {i} has invalid action '{rule['action']}'. Must be 'move' or 'copy'"
                })
        
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(rules, key=lambda x: x.get("priority", 0), reverse=True)
        
        result = {
            "status": "success",
            "source": config_uri,
            "rule_count": len(sorted_rules),
            "rules": sorted_rules
        }
        
        logger.debug(f"Successfully loaded and parsed {len(sorted_rules)} rules")
        return json.dumps(result)
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return json.dumps({"error": f"Failed to parse YAML config: {str(e)}"})
    except Exception as e:
        logger.error(f"Error loading rules from '{config_uri}': {e}")
        return json.dumps({"error": f"Failed to load rules: {str(e)}"})

# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to streamable-http)
def run_server():
    "Run the MCP server"
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Cloud Storage MCP Server on {host}:{port} with transport '{transport}'")
    mcp.run(transport=transport, host=host, port=port)

if __name__ == "__main__":
    if GCP_SERVICE_ACCOUNT_KEY is None:
        logger.warning("Please configure the GCP_SERVICE_ACCOUNT_KEY environment variable before running the server")
    else:
        logger.info("Configured GCP_SERVICE_ACCOUNT_KEY environment variable")
        
        if DEFAULT_BUCKET is None:
            logger.warning("DEFAULT_BUCKET environment variable not set - bucket must be specified in URIs")
        else:
            logger.info(f"Using DEFAULT_BUCKET: {DEFAULT_BUCKET}")
        
        # Check if JWT auth is configured
        if JWKS_URI is None:
            logger.info("No JWKS_URI configured; JWT validation disabled")
            logger.info("Starting Cloud Storage MCP Server")
            run_server()
        else:
            logger.info(f"JWT validation enabled with JWKS_URI: {JWKS_URI}")
            logger.info("Starting Cloud Storage MCP Server")
            run_server()
