
import os

def get_config():
    openai_api_key = os.getenv("OPENAI_KEY")
    azure_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_api_version = os.getenv("API_VERSION")
    azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    config_openai = {
        'api_key': openai_api_key,
    }

    config_azure = {
        'azure_endpoint': azure_api_endpoint,
        'azure_deployment': azure_api_deployment_id,
        'api_version': azure_api_version,
        'api_key': azure_api_key
    }

    return {
        "config_openai": config_openai,
        "config_azure": config_azure,
        "pinecone_api_key": pinecone_api_key,
    }