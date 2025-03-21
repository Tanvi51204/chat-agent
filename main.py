from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
from tools import search_tool, wiki_tool, save_tool

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.2-11b-vision-preview", api_key=groq_api_key)

class ResearchResponse(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")

raw_response = agent_executor.invoke({"query": query})
print("RAW RESPONSE:", raw_response)




try:
    output_data = json.loads(raw_response.get("output"))  # Convert string to dict
    
    # If "properties" key exists, extract its content
    if "properties" in output_data:
        output_data = output_data["properties"]

    structured_response = ResearchResponse(**output_data)  # Use Pydantic to parse
    print(structured_response)

except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response -", raw_response)

# response = llm.invoke("Describe the significance of LLaMA 3.2 in AI advancements.")
# print(response)