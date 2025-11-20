import os
import logging
from pydantic import BaseModel, Field
from langchain_community.tools import StructuredTool
from atlassian import Confluence
from dotenv import load_dotenv
import sys
# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------
CONFLUENCE_URL = os.getenv("conf-url")
CONFLUENCE_API_KEY = os.getenv("conf-key")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")


confluence = Confluence(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    password=CONFLUENCE_API_KEY,
    cloud=True
)

logging.basicConfig(level=logging.INFO)

# =====================================================================
# 1. READ PAGE
# =====================================================================

class ReadPageInput(BaseModel):
    page_id: str = Field(description="The Confluence page ID to read.")

def read_confluence_page(page_id: str) -> str:
    try:
        logging.info(f"Attempting to read page with ID: {page_id}")
        page = confluence.get_page_by_id(page_id, expand='body.storage')

        if page and 'body' in page and 'storage' in page['body']:
            content = page['body']['storage']['value']
            title = page['title']

            logging.info(f"Successfully read page '{title}' (ID: {page_id}).")
            return f"Success: Content of page '{title}' (ID: {page_id}):\n\n{content}"
        else:
            logging.warning(f"Page with ID {page_id} found, but it has no content.")
            return f"Warning: Page with ID {page_id} appears to be empty."

    except Exception as e:
        logging.error(f"Failed to read page with ID {page_id}. Error: {e}")
        return f"Error: Could not read Confluence page with ID {page_id}. Details: {e}"

read_confluence_tool = StructuredTool.from_function(
    func=read_confluence_page,
    name="read_confluence_page",
    description="Read the content of a specific Confluence page.",
    args_schema=ReadPageInput
)

# =====================================================================
# 2. UPDATE PAGE
# =====================================================================

class UpdatePageInput(BaseModel):
    page_id: str = Field(description="The ID of the page to update.")
    new_content: str = Field(description="The new HTML content to replace the page body.")

def update_confluence_page(page_id: str, new_content: str) -> str:
    try:
        logging.info(f"Attempting to update page with ID: {page_id}")
        current_page = confluence.get_page_by_id(page_id)

        if not current_page:
            return f"Error: Page with ID {page_id} not found. Cannot update."

        page_title = current_page['title']

        confluence.update_page(
            page_id=page_id,
            title=page_title,
            parent_id=current_page.get('parent_id'),
            body=new_content,
            minor_edit=True
        )

        logging.info(f"Successfully updated page '{page_title}' (ID: {page_id}).")
        return f"Success: Page with ID {page_id} has been updated."

    except Exception as e:
        logging.error(f"Failed to update page with ID {page_id}. Error: {e}")
        return f"Error: Could not update Confluence page with ID {page_id}. Details: {e}"

update_confluence_tool = StructuredTool.from_function(
    func=update_confluence_page,
    name="update_confluence_page",
    description="Update an existing Confluence page with new HTML content.",
    args_schema=UpdatePageInput
)

# =====================================================================
# 3. CREATE PAGE
# =====================================================================

class CreatePageInput(BaseModel):
    parent_page_id: str = Field(description="Parent page ID under which new page is created.")
    title: str = Field(description="Title of the new Confluence page.")
    content: str = Field(description="HTML content of the new page.")

def create_confluence_page(parent_page_id: str, title: str, content: str) -> str:
    try:
        logging.info(f"Fetching details for parent page ID: {parent_page_id}")

        parent_page_details = confluence.get_page_by_id(parent_page_id, expand='space')

        if not parent_page_details:
            return f"Error: Parent page with ID {parent_page_id} not found. Cannot create child page."

        space_key = parent_page_details['space']['key']
        logging.info(f"Space key determined: {space_key}")

        logging.info(f"Attempting to create new page '{title}' in space '{space_key}'.")

        response = confluence.create_page(
            space=space_key,
            title=title,
            parent_id=parent_page_id,
            body=content
        )

        new_page_id = response['id']
        link = response['_links']['webui']

        logging.info(f"Successfully created page '{title}' (ID: {new_page_id}).")
        return f"Success: New page '{title}' created with ID {new_page_id}. URL: {CONFLUENCE_URL}{link}"

    except KeyError:
        return (
            f"Error: Failed to determine Confluence space for parent ID {parent_page_id}. "
            "Ensure the parent page exists and has a valid space."
        )
    except Exception as e:
        logging.error(f"Failed to create page '{title}'. Error: {e}")
        return f"Error: Could not create page under parent {parent_page_id}. Details: {e}"

create_confluence_tool = StructuredTool.from_function(
    func=create_confluence_page,
    name="create_confluence_page",
    description="Create a new Confluence page under a specified parent page ID.",
    args_schema=CreatePageInput
)

# =====================================================================
# 4. EXPORT TOOLS FOR AGENT
# =====================================================================

confluence_tools = [
    read_confluence_tool,
    update_confluence_tool,
    create_confluence_tool
]

if __name__ == "__main__":
    resp = read_confluence_tool.invoke({"page_id" : "115310593"})
    print(resp)
    resp1 = resp.replace("Image Generation using gpt","Image Generation using gpts")
    
    update_confluence_tool.invoke({"page_id": "115310593", "new_content": resp1})
    print("Updated Successfully")