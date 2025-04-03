

# from dotenv import load_dotenv
# from pydantic import BaseModel
# from langchain_groq import ChatGroq
# import os
# # from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# import json
# from tools import search_tool, wiki_tool, save_tool

# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# llm = ChatGroq(model="llama-3.2-11b-vision-preview", api_key=groq_api_key)

# class ResearchResponse(BaseModel):
#     topic : str
#     summary : str
#     sources : list[str]
#     tools_used : list[str]

# parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# #Prompt Template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a research assistant that will help generate a research paper.
#             Answer the user query and use neccessary tools. 
#             Wrap the output in this format and provide no other text\n{format_instructions}
#             """,
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{query}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())


# tools = [search_tool, wiki_tool, save_tool]
# agent = create_tool_calling_agent(
#     llm=llm,
#     prompt=prompt,
#     tools=tools
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# query = input("What can i help you research? ")

# raw_response = agent_executor.invoke({"query": query})
# print("RAW RESPONSE:", raw_response)




# try:
#     output_data = json.loads(raw_response.get("output"))  # Convert string to dict
    
#     # If "properties" key exists, extract its content
#     if "properties" in output_data:
#         output_data = output_data["properties"]

#     structured_response = ResearchResponse(**output_data)  # Use Pydantic to parse
#     print(structured_response)

# except Exception as e:
#     print("Error parsing response:", e)
#     print("Raw Response -", raw_response)

# response = llm.invoke("Describe the significance of LLaMA 3.2 in AI advancements.")
# print(response)



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
from tools import search_tool, wiki_tool, save_tool, ResearchResponse, _save_raw_text

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# llama-3.1-70b-versatile maybe better for following structure
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, api_key=groq_api_key)

# ResearchResponse Pydantic model defined in tools.py is imported

# --- Parser Setup ---
# The PydanticOutputParser helps guide the LLM
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
# An OutputFixingParser can sometimes help if the LLM messes up the format slightly
# output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm) # Optional

# --- Prompt Template ---
# Refined prompt: Emphasize using tools and sticking to the format.
# Added {intermediate_steps} placeholder which is often used by tool calling agents
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

            If you use the `save_research_to_file` tool, make sure you provide it with the completed JSON data as a string argument. Use this saving tool *only* as the very final step if explicitly asked or if it makes sense to persist the result.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        # Use 'intermediate_steps' which is standard for AgentExecutor scratchpad
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# --- Agent Setup ---
tools = [search_tool, wiki_tool, save_tool]

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
    return_intermediate_steps=False # Usually set to False unless debugging steps
    )

# --- Execution and Formatting ---
query = input("What can I help you research? ")

print("\n--- Running Research Agent ---")
# Invoke the agent
raw_response = agent_executor.invoke({"query": query})
print("\n--- Agent Finished ---")

# --- Output Processing ---
final_output_str = raw_response.get("output")
print("\nRaw LLM Output (JSON String):")
print(repr(final_output_str))

structured_response = None
formatted_text = ""
cleaned_json_str = None # Variable to hold the cleaned string

# try:
#     # Attempt to parse the agent's final output string as JSON
#     output_data = json.loads(final_output_str)

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

# except (json.JSONDecodeError, TypeError, AttributeError) as e:
#     print("\n--- Error Parsing or Formatting Agent Output ---")
#     print(f"Error: {e}")
#     print("Could not parse the agent's output into the expected ResearchResponse structure.")
#     print("Saving the raw output instead.")
#     # Save the raw output if parsing/formatting fails
#     try:
#         _save_raw_text(f"RAW AGENT OUTPUT (Parsing Failed):\n{final_output_str}", "research_output_error.txt")
#         print("\nRaw output saved to research_output_error.txt")
#     except Exception as save_e:
#         print(f"Error saving raw output after parsing failure: {save_e}")
# except Exception as e: # Catch any other unexpected errors
#     print(f"\n--- An Unexpected Error Occurred ---")
#     print(f"Error: {e}")
#     print("Raw agent output:", final_output_str)


try:
    # Check if the output string is valid before proceeding
    if not isinstance(final_output_str, str) or not final_output_str.strip():
        raise ValueError("Agent output is empty or not a string.")

    # --- Attempt to extract JSON from the raw output string ---
    # Regex to find a JSON object potentially wrapped in markdown fences
    # re.DOTALL makes '.' match newline characters as well
    match = re.search(r"```(?:json)?\s*({.*?})\s*```|({.*?})", final_output_str, re.DOTALL)

    if match:
        # The pattern tries to capture JSON within fences (group 1)
        # or raw JSON without fences (group 2)
        cleaned_json_str = match.group(1) if match.group(1) else match.group(2)
        print("\nCleaned JSON String (extracted for parsing):")
        print(cleaned_json_str)
    else:
        # If no JSON object structure is found by regex
        raise ValueError("Could not find JSON object structure in the agent output.")

    if not cleaned_json_str:
         raise ValueError("Extracted JSON string is empty.")

    # Attempt to parse the *cleaned* JSON string
    output_data = json.loads(cleaned_json_str)

    # Handle potential nesting like {"properties": {...}} if the LLM adds it
    if isinstance(output_data, dict) and "properties" in output_data and isinstance(output_data["properties"], dict):
        output_data = output_data["properties"]

    # Validate and structure the data using the Pydantic model
    structured_response = ResearchResponse(**output_data)

    # --- Format the Pydantic Object for Readability ---
    print("\n--- Formatted Research Output ---")
    formatted_text += f"## Research Topic: {structured_response.topic}\n\n"
    formatted_text += f"### Summary:\n{structured_response.summary}\n\n"

    formatted_text += f"### Sources:\n"
    if structured_response.sources:
        for i, source in enumerate(structured_response.sources, 1):
            formatted_text += f"- {source}\n" # Use bullet points
    else:
        formatted_text += "- No specific sources listed by the agent.\n"
    formatted_text += "\n" # Add spacing

    formatted_text += f"### Tools Used by Agent:\n"
    if structured_response.tools_used:
        for i, tool_name in enumerate(structured_response.tools_used, 1):
            formatted_text += f"- {tool_name}\n" # Use bullet points
    else:
        formatted_text += "- No tools reported by the agent.\n"

    print(formatted_text)

    # --- Save the Formatted Output ---
    try:
        # Use the basic save function directly with the *formatted* text
        save_confirmation = _save_raw_text(formatted_text, "research_output_final.txt")
        print(f"\nFormatted output saved to research_output_final.txt")
    except Exception as save_e:
        print(f"\nError saving formatted output: {save_e}")

# Catch JSON errors, TypeErrors (like parsing None), AttributeErrors, and ValueErrors from our checks
except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
    print("\n--- Error Parsing or Formatting Agent Output ---")
    print(f"Error: {e}")
    print("Could not parse the agent's output into the expected ResearchResponse structure.")
    print("Saving the raw output instead.")
    # Save the raw output if parsing/formatting fails
    try:
        # Save the original raw string if cleaning/parsing failed
        raw_content_to_save = final_output_str if isinstance(final_output_str, str) else str(raw_response)
        _save_raw_text(f"RAW AGENT OUTPUT (Parsing/Cleaning Failed):\n{raw_content_to_save}", "research_output_error.txt")
        print("\nRaw output saved to research_output_error.txt")
    except Exception as save_e:
        print(f"Error saving raw output after parsing failure: {save_e}")
except Exception as e: # Catch any other unexpected errors
    print(f"\n--- An Unexpected Error Occurred ---")
    print(f"Error: {e}")
    print("Raw agent output (if available):", final_output_str)