# from dotenv import load_dotenv
# from pydantic import BaseModel
# from langchain_groq import ChatGroq
# import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# import json
# import re 
# # Import tools AND the Pydantic model AND the raw save function
# from tools import search_tool, wiki_tool, save_tool, ResearchResponse, _save_raw_text

# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# # llama-3.1-70b-versatile maybe better for following structure
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, api_key=groq_api_key)

# # ResearchResponse Pydantic model defined in tools.py is imported

# # --- Parser Setup ---
# # The PydanticOutputParser helps guide the LLM
# parser = PydanticOutputParser(pydantic_object=ResearchResponse)
# # An OutputFixingParser can sometimes help if the LLM messes up the format slightly
# # output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm) # Optional

# # --- Prompt Template ---
# # Refined prompt: Emphasize using tools and sticking to the format.
# # Added {intermediate_steps} placeholder which is often used by tool calling agents
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are an expert research assistant. Your goal is to gather information on the user's query using the available tools (search, wikipedia_search) and compile a structured response.

#             Follow these steps:
#             1. Understand the user's query.
#             2. Decide which tool(s) are needed (search for current info, wikipedia_search for encyclopedic facts). Use the tools one by one if necessary.
#             3. Synthesize the information gathered from the tools into a concise summary.
#             4. List the sources used (e.g., URLs from search, Wikipedia page title).
#             5. List the names of the tools you actually used.
#             6. Format your final answer *strictly* as a JSON object matching the following schema. Provide *only* the JSON object and nothing else.

#             Schema:
#             {format_instructions}

#             If you use the `save_research_to_file` tool, make sure you provide it with the completed JSON data as a string argument. Use this saving tool *only* as the very final step if explicitly asked or if it makes sense to persist the result.
#             """,
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{query}"),
#         # Use 'intermediate_steps' which is standard for AgentExecutor scratchpad
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())


# # --- Agent Setup ---
# tools = [search_tool, wiki_tool, save_tool]

# # Use the standard tool calling agent setup
# agent = create_tool_calling_agent(
#     llm=llm,
#     prompt=prompt,
#     tools=tools
# )

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors="Check your output and make sure it conforms to the format instructions.", # Nicer error handling
#     return_intermediate_steps=False # Usually set to False unless debugging steps
#     )

# # --- Execution and Formatting ---
# query = input("What can I help you research? ")

# print("\n--- Running Research Agent ---")
# # Invoke the agent
# raw_response = agent_executor.invoke({"query": query})
# print("\n--- Agent Finished ---")

# # --- Output Processing ---
# final_output_str = raw_response.get("output")
# print("\nRaw LLM Output (JSON String):")
# print(repr(final_output_str))

# structured_response = None
# formatted_text = ""
# cleaned_json_str = None # Variable to hold the cleaned string

# try:
#     # Check if the output string is valid before proceeding
#     if not isinstance(final_output_str, str) or not final_output_str.strip():
#         raise ValueError("Agent output is empty or not a string.")

#     # --- Attempt to extract JSON from the raw output string ---
#     # Regex to find a JSON object potentially wrapped in markdown fences
#     # re.DOTALL makes '.' match newline characters as well
#     match = re.search(r"```(?:json)?\s*({.*?})\s*```|({.*?})", final_output_str, re.DOTALL)

#     if match:
#         # The pattern tries to capture JSON within fences (group 1)
#         # or raw JSON without fences (group 2)
#         cleaned_json_str = match.group(1) if match.group(1) else match.group(2)
#         print("\nCleaned JSON String (extracted for parsing):")
#         print(cleaned_json_str)
#     else:
#         # If no JSON object structure is found by regex
#         raise ValueError("Could not find JSON object structure in the agent output.")

#     if not cleaned_json_str:
#          raise ValueError("Extracted JSON string is empty.")

#     # Attempt to parse the *cleaned* JSON string
#     output_data = json.loads(cleaned_json_str)

#     # Handle potential nesting like {"properties": {...}} if the LLM adds it
#     if isinstance(output_data, dict) and "properties" in output_data and isinstance(output_data["properties"], dict):
#         output_data = output_data["properties"]

#     # Validate and structure the data using the Pydantic model
#     structured_response = ResearchResponse(**output_data)

#     # --- Format the Pydantic Object for Readability ---
#     print("\n--- Formatted Research Output ---")
#     formatted_text += f"## Research Topic: {structured_response.topic}\n\n"
#     formatted_text += f"### Summary:\n{structured_response.summary}\n\n"

#     formatted_text += f"### Sources:\n"
#     if structured_response.sources:
#         for i, source in enumerate(structured_response.sources, 1):
#             formatted_text += f"- {source}\n" # Use bullet points
#     else:
#         formatted_text += "- No specific sources listed by the agent.\n"
#     formatted_text += "\n" # Add spacing

#     formatted_text += f"### Tools Used by Agent:\n"
#     if structured_response.tools_used:
#         for i, tool_name in enumerate(structured_response.tools_used, 1):
#             formatted_text += f"- {tool_name}\n" # Use bullet points
#     else:
#         formatted_text += "- No tools reported by the agent.\n"

#     print(formatted_text)

#     # --- Save the Formatted Output ---
#     try:
#         # Use the basic save function directly with the *formatted* text
#         save_confirmation = _save_raw_text(formatted_text, "research_output_final.txt")
#         print(f"\nFormatted output saved to research_output_final.txt")
#     except Exception as save_e:
#         print(f"\nError saving formatted output: {save_e}")

# # Catch JSON errors, TypeErrors (like parsing None), AttributeErrors, and ValueErrors from our checks
# except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
#     print("\n--- Error Parsing or Formatting Agent Output ---")
#     print(f"Error: {e}")
#     print("Could not parse the agent's output into the expected ResearchResponse structure.")
#     print("Saving the raw output instead.")
#     # Save the raw output if parsing/formatting fails
#     try:
#         # Save the original raw string if cleaning/parsing failed
#         raw_content_to_save = final_output_str if isinstance(final_output_str, str) else str(raw_response)
#         _save_raw_text(f"RAW AGENT OUTPUT (Parsing/Cleaning Failed):\n{raw_content_to_save}", "research_output_error.txt")
#         print("\nRaw output saved to research_output_error.txt")
#     except Exception as save_e:
#         print(f"Error saving raw output after parsing failure: {save_e}")
# except Exception as e: # Catch any other unexpected errors
#     print(f"\n--- An Unexpected Error Occurred ---")
#     print(f"Error: {e}")
#     print("Raw agent output (if available):", final_output_str)


# main.py
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
import re

# Import tools AND the Pydantic model AND the raw save function
# Removed save_tool import
from tools import search_tool, wiki_tool, ResearchResponse, _save_content_to_file # Updated import

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# Consider trying llama-3.1-70b-versatile if needed
# Model updated based on previous context, ensure it's available/suitable
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, api_key=groq_api_key)
# Using llama3.1 70b as it might be better with complex instructions/tool use
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, api_key=groq_api_key)


# ResearchResponse Pydantic model defined in tools.py is imported

# --- Parser Setup ---
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# --- Prompt Template ---
# Refined prompt: Emphasize using tools and sticking to the format.
# Removed mention of save_tool as saving happens after agent execution.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert research assistant. Your goal is to gather information on the user's query using the available tools (search, wikipedia_search) and compile a structured response.

            Follow these steps:
            1. Understand the user's query.
            2. Decide which tool(s) are needed (search for current info, wikipedia_search for encyclopedic facts). Use the tools one by one if necessary.
            3. Synthesize the information gathered from the tools into a concise summary.
            4. List the sources used (e.g., URLs from search, Wikipedia page title).
            5. List the names of the tools you actually used.
            6. Format your final answer *strictly* as a JSON object matching the following schema. Provide *only* the JSON object and nothing else.

            Schema:
            {format_instructions}

            Do not include explanations or conversational text outside the required JSON structure in your final output.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# --- Agent Setup ---
tools = [search_tool, wiki_tool] # REMOVED save_tool

# Use the standard tool calling agent setup
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms to the format instructions.", # Nicer error handling
    return_intermediate_steps=False
)

# --- Execution ---
query = input("What can I help you research? ")

print("\n--- Running Research Agent ---")
raw_response = agent_executor.invoke({"query": query})
print("\n--- Agent Finished ---")

# --- Output Processing ---
final_output_str = raw_response.get("output")
print("\nRaw LLM Output (JSON String):")
print(repr(final_output_str))

structured_response = None
html_output = "" # Variable to hold the formatted HTML
cleaned_json_str = None

# --- Basic HTML Styling ---
# Added some simple CSS for better readability in the HTML output
html_style = """
<style>
  body { font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }
  h2 { color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
  h3 { color: #555; margin-top: 25px; }
  ul { list-style-type: disc; margin-left: 20px; }
  li { margin-bottom: 5px; }
  p { margin-bottom: 15px; }
  .error { color: red; font-weight: bold; }
  .raw-output { background-color: #f0f0f0; border: 1px solid #ddd; padding: 10px; white-space: pre-wrap; word-wrap: break-word; margin-top: 20px; }
</style>
"""

try:
    # Check if the output string is valid before proceeding
    if not isinstance(final_output_str, str) or not final_output_str.strip():
        raise ValueError("Agent output is empty or not a string.")

    # --- Attempt to extract JSON from the raw output string ---
    match = re.search(r"```(?:json)?\s*({.*?})\s*```|({.*?})", final_output_str, re.DOTALL)

    if match:
        cleaned_json_str = match.group(1) if match.group(1) else match.group(2)
        print("\nCleaned JSON String (extracted for parsing):")
        print(cleaned_json_str)
    else:
        raise ValueError("Could not find JSON object structure in the agent output.")

    if not cleaned_json_str:
         raise ValueError("Extracted JSON string is empty.")

    # Attempt to parse the *cleaned* JSON string
    output_data = json.loads(cleaned_json_str)

    # Handle potential nesting (optional, depends on LLM)
    if isinstance(output_data, dict) and "properties" in output_data and isinstance(output_data["properties"], dict):
        output_data = output_data["properties"]

    # Validate and structure the data using the Pydantic model
    structured_response = ResearchResponse(**output_data)

    # --- Format the Pydantic Object into HTML ---
    print("\n--- Generating HTML Output ---")
    html_output += f"<!DOCTYPE html>\n<html>\n<head>\n"
    html_output += f"<title>Research: {structured_response.topic}</title>\n"
    html_output += f"<meta charset=\"UTF-8\">\n" # Ensure proper encoding
    html_output += html_style # Add the CSS styles
    html_output += "</head>\n<body>\n"

    html_output += f"<h2>Research Topic: {structured_response.topic}</h2>\n"

    html_output += f"<h3>Summary:</h3>\n<p>{structured_response.summary}</p>\n" # Wrap summary in <p>

    html_output += f"<h3>Sources:</h3>\n"
    if structured_response.sources:
        html_output += "<ul>\n"
        for source in structured_response.sources:
             # Basic check if source looks like a URL to make it clickable
             if source.startswith("http://") or source.startswith("https://"):
                 html_output += f"  <li><a href=\"{source}\" target=\"_blank\">{source}</a></li>\n"
             else:
                 html_output += f"  <li>{source}</li>\n" # Keep non-URLs as text
        html_output += "</ul>\n"
    else:
        html_output += "<p>No specific sources listed by the agent.</p>\n"

    html_output += f"<h3>Tools Used by Agent:</h3>\n"
    if structured_response.tools_used:
        html_output += "<ul>\n"
        for tool_name in structured_response.tools_used:
            html_output += f"  <li>{tool_name}</li>\n"
        html_output += "</ul>\n"
    else:
        html_output += "<p>No tools reported by the agent.</p>\n"

    html_output += "</body>\n</html>"

    print("HTML content generated successfully.")
    # print(html_output) # Optional: print HTML to console

    # --- Save the Formatted HTML Output ---
    output_filename = "research_output_final.html"
    try:
        _save_content_to_file(html_output, output_filename)
        print(f"\nFormatted HTML output saved to {output_filename}")
        print(f"You can open this file in your web browser.")
    except Exception as save_e:
        print(f"\nError saving formatted HTML output: {save_e}")

# Catch JSON errors, TypeErrors (like parsing None), AttributeErrors, and ValueErrors from our checks
except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
    print("\n--- Error Parsing or Formatting Agent Output ---")
    print(f"Error: {e}")
    print("Could not parse the agent's output or generate HTML.")
    print("Saving the raw output instead to a text file.")
    # Save the raw output if parsing/formatting fails
    error_filename = "research_output_error.log" # Use .log for errors
    try:
        raw_content_to_save = final_output_str if isinstance(final_output_str, str) else str(raw_response)
        # Generate a simple error HTML page containing the raw output
        error_html = f"""<!DOCTYPE html>
<html>
<head><title>Agent Output Error</title>{html_style}</head>
<body>
<h2>Agent Processing Error</h2>
<p class="error">Failed to parse JSON or format output. Error: {e}</p>
<h3>Raw Agent Output:</h3>
<div class="raw-output">{raw_content_to_save}</div>
</body>
</html>"""
        # Save the error HTML
        # _save_content_to_file(error_html, "research_output_error.html")
        # print(f"\nError details and raw output saved to research_output_error.html")

        # Alternatively, save raw text to a log file:
        _save_content_to_file(f"ERROR: {e}\nRAW AGENT OUTPUT:\n{raw_content_to_save}", error_filename)
        print(f"\nRaw output saved to {error_filename}")

    except Exception as save_e:
        print(f"Error saving raw output after parsing failure: {save_e}")

except Exception as e: # Catch any other unexpected errors
    print(f"\n--- An Unexpected Error Occurred ---")
    print(f"Error: {e}")
    error_filename = "research_output_unexpected_error.log"
    try:
        raw_content_to_save = final_output_str if isinstance(final_output_str, str) else str(raw_response)
        _save_content_to_file(f"UNEXPECTED ERROR: {e}\nRAW AGENT OUTPUT (if available):\n{raw_content_to_save}", error_filename)
        print(f"Error details and raw output (if available) saved to {error_filename}")
    except Exception as save_e:
        print(f"Additionally, failed to save error log: {save_e}")