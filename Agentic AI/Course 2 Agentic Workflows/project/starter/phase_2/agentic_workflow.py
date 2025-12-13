# Agentic Workflow
# imports and envs
# Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load the openai_api_key variable with your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Instantiate Agents
# - Action Planning Agent
# - Product Manager Knowledge Agent
# - Product Manager Evaluation Agent
# - Program Manager Knowledge Agent
# - Program Manager Evaluation Agent
# - Develeopment Engineer Knowledge Agent
# - Develeopment Engineer Evaluation Agent

### Action Planning Agent
knowledge_action_planning ="""
Extract exactly 3 high-level workflow steps for Email Router project planning:
1. Generate user stories - create comprehensive user stories for the Email Router product
2. Define product features - create product features with Feature Name, Description, Key Functionality, and User Benefit\n"
3. Create development tasks - create engineering tasks with Task ID, Title, Description, Acceptance Criteria, Estimated Effort, and Dependencies\n"
Focus specifically on Email Router functionality, not generic examples.
"""

# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key = openai_api_key,
    knowledge = knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = """
You are a Product Manager.
You, as a product manger, are responsible for defining the user stories for a product.

Return 3-4 user stories in the form:
1: As a [user type], I want [action/feature] so that [benefit/jobs to be done].
2: As a [user type], I want [action/feature] so that [benefit/jobs to be done].
3: As a [user type], I want [action/feature] so that [benefit/jobs to be done].
(continue as many as you need)

Do not include any functionality outside this product.

Specification:
[PASTE EMAIL ROUTER SPEC HERE]
"""

path_to_document_file = os.path.join(os.path.abspath("."), "Product-Spec-Email-Router.txt")
product_spec = open(path_to_document_file, "rt").read()
knowledge_product_manager = f"""
Stories are defined by writing sentences with a persona, an action, and a desired outcome.
The sentences always start with: As a
Write several stories for the product spec based on your knowledge.
The personas are the different users of the product.

Your knowledge is defined as below:
{product_spec}
"""

# Instantiate a product_manager_knowledge_agent using 'persona_product_manager'
# and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona_product_manager,
    knowledge = knowledge_product_manager
)

# Product Manager - Evaluation Agent
# Define the persona and evaluation criteria for a Product Manager evaluation agent
# and instantiate it as product_manager_evaluation_agent.
# This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories
# (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").

persona_product_manager_evaluation_agent = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager_evaluatoin_agent = "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key = openai_api_key,
    persona = persona_product_manager_evaluation_agent,
    evaluation_criteria = evaluation_criteria_product_manager_evaluatoin_agent,
    worker_agent = product_manager_knowledge_agent,
    max_interactions=3
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = """
You are a Program Manager, you are responsible for defining feature from user stories.
"""

knowledge_program_manager = """
Your output consists only of list of features.
Features are created from user stories step by step as below:
step 1. Find a user story statement from the Previous output.
step 2. Find a type of user after "As a" statement in the user story.
step 3. Find an action or feature after "I want" statement in the user story..
step 4. Find benefit/value after "so that" statement in the user story..
step 5. iterate step 1 to step 4 until all user stories are picked up.
step 6. Create features in structure below from the user stories.
- Feature Name: [Insert a clear, concise title that identifies the capability]
- Description: [Insert a brief explanation of what the feature does and its purpose]
- Key Functionality: [Insert the specific capabilities or actions the feature provides]
- User Benefit: [Insert how this feature creates value for the user]
- User Story: [Insert a user story that describes how the feature will be used]

The user stories are given in the user prompt after a line which includes "Previous ouput:".
1: As a [type of user], I want [an action or feature] so that [benefit/value]
2: As a [type of user], I want [an action or feature] so that [benefit/value]
3: As a [type of user], I want [an action or feature] so that [benefit/value]
(continues)
"""

# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key= openai_api_key,
    persona = persona_program_manager,
    knowledge = knowledge_program_manager
)


# Program Manager - Evaluation Agent

# Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
persona_program_manager_eval = """
You are an evaluation agent that checks the answers of other worker agents.
"""
evaluation_criteria_program_manager_evaluation_agent = """
The output only lists product features described in the following structure:
- Feature Name: [Insert a clear, concise title that identifies the capability]
- Description: [Insert a brief explanation of what the feature does and its purpose]
- Key Functionality: [Insert the specific capabilities or actions the feature provides]
- User Benefit: [Insert how this feature creates value for the user]
- User Story: [Insert a user story that describes how the feature will be used]
"""
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key = openai_api_key,
    persona = persona_program_manager_eval,
    evaluation_criteria = evaluation_criteria_program_manager_evaluation_agent,
    worker_agent = program_manager_knowledge_agent,
    max_interactions = 3
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = """
You are a Development Engineer, you are responsible for defining development tasks from features.
"""

knowledge_dev_engineer = """

Each developmment task for the output is defined in the following format:
- Task ID: A unique identifier for the task.
- Task Title: A concise title that describes the task.
- Related User Story: The user story or requirement that the task is addressing.
- Description: A detailed explanation of what needs to be done for the task.
- Acceptance Criteria: The criteria that need to be met for the task to be considered complete.
- Estimated Effort: An estimation of the time and resources required to complete the task.
- Dependencies: Any tasks or requirements that need to be completed before this task can start.

Each feature in the user prompt is defined in the following format after "Previous output":
- Feature Name: [Insert a clear, concise title that identifies the capability]
- Description: [Insert a brief explanation of what the feature does and its purpose]
- Key Functionality: [Insert the specific capabilities or actions the feature provides]
- User Benefit: [Insert how this feature creates value for the user]
- User Story: [Insert a user story that describes how the feature will be used]

Follow the steps below to create development tasks:
step 1. Find the features in a list form within the user prompt.
step 2. Find the Feature Name of the feature found in step 1.
step 3. Find the Description of the feature found in step 1.
step 4. Find the Key Functionality of the feature found in step 1.
step 5. Find the User Benefit of the feature found in step 1.
step 6. Find the User Story of the feature found in step 1.
step 7. Generate development task from the feature description, key functionality, user benefit, and user story.
step 8. Iterate above steps for all the features in the user prompt.
"""

# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona_dev_engineer,
    knowledge = knowledge_dev_engineer,
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_development_engineer_evaluation_agent = """
The answer should be a collection of tasks:
Each task are defined in the structure below:
- Task ID: A unique identifier for the task.
- Task Title: A concise title that describes the task.
- Related User Story: The user story or requirement that the task is addressing.
- Description: A detailed explanation of what needs to be done for the task.
- Acceptance Criteria: The criteria that need to be met for the task to be considered complete.
- Estimated Effort: An estimation of the time and resources required to complete the task.
- Dependencies: Any tasks or requirements that need to be completed before this task can start.
"""

# Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key = openai_api_key,
    persona = persona_dev_engineer_eval,
    evaluation_criteria = evaluation_criteria_development_engineer_evaluation_agent,
    worker_agent = development_engineer_knowledge_agent,
    max_interactions = 5
)

# Routing Agent
routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
)

# Job function persona support functions
# Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
# Updated for reusability of the code
def support_function(query: str,
                    eval_agent: EvaluationAgent):
    # Following commented out as the evaluation agent implementation execute this
    # Rather if we call here, initial query is againg executed by a worker agent
    # in the evaluation agent.
    # See the worker agent is set in each evaluation agent, when they are intantiated.
    #initial_response = knowledge_agent.respond(query)
    evaluation_response = eval_agent.evaluate(query)
    return evaluation_response["final_response"]

routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Only responsible for defining one user story of a product spec.",
        "func": lambda query: support_function(query, product_manager_evaluation_agent)
    },
    {
        "name": "Program Manager",
        "description": "Only Responsible for defining one specific product feature from a user stories",
        "func": lambda query: support_function(f"User stories are defined in {query}", program_manager_evaluation_agent)
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for define development tasks from features",
        "func": lambda query: support_function(f"Features are defined in {query}", development_engineer_evaluation_agent)
    }
]

# Run the workflow
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "Define a project tasks for this product be by creating user stories, key features, and development tasks step by step."
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

print("Action planned by the planning agent")
for step in steps:
    print(step)

#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
def exec_step(step: str):
    print(f"----------------------\nstep:\n----------------------\n{step}\n----------------------\n")
    result = routing_agent.route(step)
    print(f"----------------------\nresult:\n----------------------\n{result}\n----------------------\n")
    return result

completed_steps = [exec_step(step) for step in steps]

#   4. After the loop, print the final output of the workflow (the last completed step).
print(f"~~~~~~~~~~~~~~~~~~~\nfinal output\n~~~~~~~~~~~~~~~~~~~\n ")
for step, completed_step in zip(steps, completed_steps):
    print(f"----------------------\nstep: {step}\n----------------------\n")
    print(completed_step + "\n\n")
print(f"~~~~~~~~~~~~~~~~~~~\n ")


