from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from typing import Tuple
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
st.title( "Star Wars Character Analyzer")
def star_wars_info(name: str) -> Tuple[str, str]:
    """
    Function to get information about a Star Wars character using OpenAI's API.
    :param name: Name of the Star Wars character.
    :return: Tuple containing the films the character's appeared in.
    """
    # Load environment variables from .env file

    summary_template = """
    given the name of a Star Wars character {name} I want you to create:
    1. A short summary
    2. a list of the films they've appeared in.
    """
    # Define the prompt template for the LLM
    prompt_template = PromptTemplate(
        input_variables=["name"],
        template=summary_template
    )

    # Initialize the language model with specific parameters
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o-mini",
        max_tokens=1000,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create an LLM chain with the prompt and language model
    chain = prompt_template | llm 
    # Run the chain with the provided name and get the response
    response = chain.invoke({"name": name}) 
    st.info(response)



with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Which Star Wars character would you like to analyze?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        star_wars_info(text)


    # Clean up the response to extract the character's name and description
    # response = re.sub(r"^.*?\n", "", response).strip()n 
    



if __name__ == "__main__":
    load_dotenv()
    print("Starting character analyzer...")
    print(star_wars_info(name = "Luke Skywalker"))

   