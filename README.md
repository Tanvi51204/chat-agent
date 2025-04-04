
# LangChain Research Agent with HTML Output

This project implements an AI research assistant using the LangChain framework and a Groq-powered Large Language Model (LLM). It takes a user's research query, utilizes web search (DuckDuckGo) and Wikipedia tools to gather information, and then synthesizes the findings into a structured summary. The final output is saved as an HTML file for easy viewing in a web browser.

## Features

*   **AI-Powered Research:** Leverages an LLM via the Groq API for understanding queries and generating summaries.
*   **Tool Integration:** Uses LangChain Agents to dynamically choose and utilize:
    *   DuckDuckGo Search for current web information.
    *   Wikipedia Search for encyclopedic knowledge.
*   **Structured Output:** Parses the LLM's response into a predefined Pydantic model (`ResearchResponse`).
*   **HTML Formatting:** Generates a clean, readable HTML file (`research_output_final.html`) containing the research topic, summary, sources (with clickable links if applicable), and tools used.
*   **Error Handling:** Includes basic error handling for JSON parsing and saves raw agent output to `.log` files if processing fails.
*   **Environment Variable Management:** Uses `python-dotenv` to securely manage the Groq API key.

## Technology Stack

*   Python 3.8+
*   LangChain (`langchain`, `langchain-groq`, `langchain-community`)
*   Groq API (requires an API key)
*   Pydantic (for data validation and structuring)
*   `python-dotenv` (for environment variables)
*   `duckduckgo-search` (for the search tool)
*   `wikipedia` (for the Wikipedia tool)

## Usage

1.  Ensure your virtual environment is activated.
2.  Run the main script from the terminal:
    ```bash
    python main.py
    ```
3.  The script will prompt you to enter your research query:
    ```
    What can I help you research?
    ```
4.  Type your query and press Enter.
5.  The agent will process the query, potentially using the search and Wikipedia tools (verbose output showing the agent's steps will be printed to the console).
6.  Once finished, the script will attempt to parse the LLM's response and generate an HTML file.

## Output

*   **Successful Run:** A file named `research_output_final.html` will be created in the project directory. Open this file in your web browser to view the formatted research summary.
*   **Error During Processing:** If the script encounters an error while parsing the LLM's output or generating the HTML, it will print an error message to the console and save the raw, unprocessed output from the agent to a `.log` file (e.g., `research_output_error.log` or `research_output_unexpected_error.log`).
