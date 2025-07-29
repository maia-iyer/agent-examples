import requests
import os
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# set configurations from environment variables
#ISSUER_URL = os.getenv("ISSUER_URL")
#JWKS_URL = f"{ISSUER_URL}/protocol/openid-connect/certs"
#CLIENT_ID = os.getenv("CLIENT_ID")

# slack bot token to access slack api
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "YOUR_SLACK_BOT_TOKEN")
try: 
    slack_client = WebClient(token=SLACK_BOT_TOKEN)
    auth_test = slack_client.auth_test()
    print(f"Successfully authenticated as bot '{auth_test['user']}' in workspace '{auth_test['team']}'.")
except SlackApiError as e:
    # Handle authentication errors, such as an invalid token
    print(f"Error authenticating with Slack: {e.response['error']}")
    slack_client = None
except Exception as e:
    print(f"An unexpected error occurred during Slack client initialization: {e}")
    slack_client = None

mcp = FastMCP("Slack", port=8000)

@mcp.tool()
def get_channels() -> List[Dict[str, Any]]:
    if slack_client in None: 
        return [{"error": "Slack client not initialized"}]
    
    try:
        # Call the conversations_list method to get public channels
        result = slack_client.conversations_list(types="public_channel")
        channels = result.get("channels", [])
        # We'll just return some key information for each channel
        return [
            {"id": c["id"], "name": c["name"], "purpose": c.get("purpose", {}).get("value", "")}
            for c in channels
        ]
    except SlackApiError as e:
        # Handle API errors and return a descriptive message
        return [{"error": f"Slack API Error: {e.response['error']}"}]
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {e}"}]

# host can be specified with HOST env variable
def run_server():
    mcp.run(transport="sse") 

if __name__ == "__main__":
    if not slack_client or SLACK_BOT_TOKEN == "YOUR_SLACK_BOT_TOKEN":
        print("Please configure the SLACK_BOT_TOKEN environment variable before running the server")
    else:
        print("Starting Slack MCP Server")
        run_server()