from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load the openai_api_key variable with your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_prompt_agent = AugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona
)
# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_prompt_agent.respond(input_text = prompt)
# Print the agent's response
print(augmented_agent_response)

# Add a comment explaining:
print("Q: What knowledge the agent likely used to answer the prompt")
print(f"A: Training data when building the LLM model {augmented_prompt_agent.model}")
print("Q: How the system prompt specifying the persona affected the agent's response.")
print(f"""
A: It adds role and context defined in the system prompt to the LLM model response.
So it should be \"{persona}\"\n\n""")