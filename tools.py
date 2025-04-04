# # tools.py
# from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.tools import Tool
# from datetime import datetime
# import json # Import json

# # Define the Pydantic model here or import it if it's in a separate models.py
# from pydantic import BaseModel

# class ResearchResponse(BaseModel):
#     topic : str
#     summary : str
#     sources : list[str]
#     tools_used : list[str]


# def save_research_to_txt(research_data_json: str, filename: str = "research_output.txt"):
#     """
#     Parses research data JSON, formats it for readability, and saves it to a text file.
#     Expects input as a JSON string matching the ResearchResponse model.
#     """
#     try:
#         # Attempt to parse the JSON string input
#         data_dict = json.loads(research_data_json)

#         # If the structure includes 'properties' (common with some structured output), extract it
#         if isinstance(data_dict, dict) and "properties" in data_dict and isinstance(data_dict["properties"], dict):
#              data_dict = data_dict["properties"]
#         elif isinstance(data_dict, dict) and "topic" in data_dict: # Directly use if keys match
#              pass # data_dict is likely already in the correct structure
#         else:
#              # If parsing fails or structure is unexpected, try to save raw string nicely
#              print(f"Warning: Input to save_tool wasn't the expected JSON structure. Saving raw content.")
#              _save_raw_text(research_data_json, filename)
#              return f"Raw data saved to {filename} due to unexpected format."

#         # Validate with Pydantic (optional but good practice)
#         research_response = ResearchResponse(**data_dict)

#         # Format the output
#         formatted_text = f"## Research Topic: {research_response.topic}\n\n"
#         formatted_text += f"### Summary:\n{research_response.summary}\n\n"

#         formatted_text += f"### Sources:\n"
#         if research_response.sources:
#             for i, source in enumerate(research_response.sources, 1):
#                 formatted_text += f"{i}. {source}\n"
#         else:
#             formatted_text += "- No sources listed.\n"
#         formatted_text += "\n"

#         formatted_text += f"### Tools Used:\n"
#         if research_response.tools_used:
#             for i, tool in enumerate(research_response.tools_used, 1):
#                 formatted_text += f"{i}. {tool}\n"
#         else:
#              formatted_text += "- No tools listed.\n"

#         _save_raw_text(formatted_text, filename) # Use helper to save
#         return f"Formatted research on '{research_response.topic}' saved to {filename}"

#     except json.JSONDecodeError:
#         print(f"Warning: Input to save_tool was not valid JSON. Saving raw content.")
#         _save_raw_text(research_data_json, filename)
#         return f"Raw data saved to {filename} as input was not valid JSON."
#     except Exception as e:
#         print(f"Error processing or saving data: {e}")
#         _save_raw_text(f"Error during processing: {e}\nRaw Data:\n{research_data_json}", filename)
#         return f"Error saving data to {filename}. See file for details."


# def _save_raw_text(content: str, filename: str):
#     """Helper function to append text content with a timestamp."""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     header = f"\n--- Research Output Entry ---\nTimestamp: {timestamp}\n---\n"
#     full_content = header + content + "\n---\n"

#     try:
#         with open(filename, "a", encoding="utf-8") as f:
#             f.write(full_content)
#     except Exception as e:
#         print(f"Failed to write to file {filename}: {e}")


# # --- Tool Definitions ---

# save_tool = Tool(
#     name="save_research_to_file",
#     func=save_research_to_txt,
#     description="Parses structured research data (JSON string with keys: topic, summary, sources, tools_used), formats it, and saves it to a text file. Use this as the final step to save the completed research.",
#     # Ensure the description guides the LLM to pass the correct JSON structure string
# )

# search = DuckDuckGoSearchRun()
# search_tool = Tool(
#     name="search",
#     func=search.run,
#     description="Search the web for current information about a topic. Use this for recent events or general web searches.",
# )

# # Increased max chars for potentially more useful summaries
# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
# wiki_tool = Tool(
#     name="wikipedia_search", # More specific name
#     func=api_wrapper.run,   # Use the wrapper's run method directly
#     description="Search Wikipedia for encyclopedic information about a specific entity, concept, or historical event. Provides a summary from a Wikipedia page."
# )

# tools.py
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json

# Define the Pydantic model here or import it if it's in a separate models.py
from pydantic import BaseModel

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Keep this basic helper function for saving content (HTML or raw text)
def _save_content_to_file(content: str, filename: str):
    """Helper function to write or append text content to a file."""
    mode = "w" if filename.endswith(".html") else "a" # Overwrite HTML, append logs
    encoding = "utf-8"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(filename, mode, encoding=encoding) as f:
            if mode == "a": # Add timestamp header for appended log files
                 header = f"\n--- Log Entry ---\nTimestamp: {timestamp}\n---\n"
                 f.write(header)
            f.write(content)
            if mode == "a":
                 f.write("\n---\n") # Footer for appended log entries
    except Exception as e:
        print(f"Failed to write to file {filename}: {e}")
    # Return a confirmation message (optional, used internally now)
    # return f"Content saved to {filename}" # Not needed if called directly


# --- Tool Definitions ---
# REMOVED: save_tool as saving is handled in main.py after processing

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for current information about a topic. Use this for recent events or general web searches.",
)

# Increased max chars for potentially more useful summaries
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = Tool(
    name="wikipedia_search", # More specific name
    func=api_wrapper.run,   # Use the wrapper's run method directly
    description="Search Wikipedia for encyclopedic information about a specific entity, concept, or historical event. Provides a summary from a Wikipedia page."
)

# Export ONLY the necessary items if importing elsewhere
# __all__ = ['search_tool', 'wiki_tool', 'ResearchResponse', '_save_content_to_file']