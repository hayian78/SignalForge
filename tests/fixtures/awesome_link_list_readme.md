Adala: Autonomous Data (Labeling) Agent framework

<details>

![Image](https://github.com/HumanSignal/Adala/raw/master/docs/src/img/logo-dark-mode.png)

### Category
General purpose, Build your own, Multi-agent

### Description

- **Reliable agents**: Built on ground truth data for consistent, trustworthy results.
- **Controllable output**: Tailor output with flexible constraints to fit your needs.
- **Specialized in data processing**: Agents excel in custom data labeling and processing tasks.
- **Autonomous learning**: Agents evolve through observations and reflections, not just automation.
- **Flexible and extensible runtime**: Adaptable framework with community-driven evolution for diverse needs.
- **Easily customizable**: Develop agents swiftly for unique challenges, no steep learning curve.

### Links
- [Documentation](https://humansignal.github.io/Adala/) 
- [Discord](https://discord.gg/QBtgTbXTgU)
- [GitHub](https://github.com/HumanSignal/Adala)
</details>

## [Agent4Rec](https://github.com/LehengTHU/Agent4Rec)
Recommender system simulator with 1,000 agents

<details>
<p><img src="https://github.com/LehengTHU/Agent4Rec/raw/master/assets/sandbox.png" alt="Image" /></p>

### Category
General purpose, Build your own, Multi-agent

### Description
- Agent4Rec is a recommender system simulator that utilizes 1,000 LLM-empowered generative agents.
- These agents are initialized from the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) dataset, embodying varied social traits and preferences.
- Each agent interacts with personalized movie recommendations in a page-by-page manner and undertakes various actions such as watching, rating, evaluating, exiting, and interviewing. 

### Links
- [Paper](https://arxiv.org/abs/2310.10108)

</details>

## [AgentForge](https://github.com/DataBassGit/AgentForge)
LLM-agnostic platform for agent building & testing

<details>

![Image](https://pbs.twimg.com/profile_images/1667167265060528129/l8S9vtP2_400x400.jpg)

### Category
General purpose, Build your own, Multi-agent

### Description
- A low-code framework designed for the swift creation, testing, and iteration of AI-powered autonomous agents and Cognitive Architectures, compatible with various LLM models.
- Facilitates building custom agents and cognitive architectures with ease.
- Supports multiple LLM models including OpenAI, Anthropic's Claude, and local Oobabooga, allowing flexibility in running different models for different agents based on specific requirements.
- Provides customizable agent memory management and on-the-fly prompt editing for rapid development and testing.
- Comes with a database-agnostic design ensuring seamless extensibility, with straightforward integration with different databases like ChromaDB for various AI projects.

### Links
- [GitHub](https://github.com/DataBassGit/AgentForge)
- [Web](https://www.agentforge.net/)
- [Discord](https://discord.com/invite/ttpXHUtCW6)
- [X](https://twitter.com/AgentForge)

</details>

## [AgentGPT](https://agentgpt.reworkd.ai/)
Browser-based no-code version of AutoGPT
<details>

![Image](https://raw.githubusercontent.com/reworkd/AgentGPT/main/next/public/banner.png)


### Category
General purpose

### Description
- A no-code platform
- Process:
	- Assigning a goal to the agent
	- Witnessing its thinking process
	- Formulation of an execution plan
	- Taking actions accordingly
- Uses OpenAI functions
- Supports gpt-3.5-16k, pinecone and pg_vector databases
- Stack
	- Frontend: NextJS + Typescript
	- Backend: FastAPI + Python
	- DB: MySQL through docker with the option of running SQLite locally

<!--
### Features
- Uses OpenAI **functions**
- Supports gpt-3.5-16k, pinecone and pg_vector databases

### Stack
- Frontend: NextJS + Typescript
- Backend: FastAPI + Python
	- DB: MySQL through docker with the option of running SQLite locally
	-->

### Links
- [Documentation](https://docs.reworkd.ai/)
- [Website](https://agentgpt.reworkd.ai/)
- [GitHub](https://github.com/reworkd/AgentGPT)
</details>

<!-- This is a comment that appears only in the raw text -->

## [AgentPilot](https://github.com/jbexta/AgentPilot)
Build, manage, and chat with agents in desktop app


<details>

![Image](https://github.com/jbexta/AgentPilot/raw/master/docs/demo.png)

### Category
General purpose

### Description

- Integrated into Open Interpreter and MemGPT
- Group chats feature



### Links
- [GitHub](https://github.com/jbexta/AgentPilot)
- [X ](https://twitter.com/AgentPilotAI)
- 
  
</details>

## [Agents](https://github.com/aiwaves-cn/agents)

Library/framework for building language agents

<details>

![Image](https://github.com/aiwaves-cn/agents/raw/master/assets/agents-logo.png)

### Category
General purpose, Build your own, Multi-agent

### Description
-   **Long-short Term Memory**: Language agents in the library are equipped with both long-term memory implemented via VectorDB + Semantic Search and short-term memory (working memory) maintained and updated by an LLM.
-   **Tool Usage**: Language agents in the library can use any external tools via  [function-calling](https://platform.openai.com/docs/guides/gpt/function-calling)  and developers can add customized tools/APIs  [here](https://github.com/aiwaves-cn/agents/blob/master/src/agents/Component/ToolComponent.py).
-   **Web Navigation**: Language agents in the library can use search engines to navigate the web and get useful information.
-   **Multi-agent Communication**: In addition to single language agents, the library supports building multi-agent systems in which language agents can communicate with other language agents and the environment. Different from most existing frameworks for multi-agent systems that use pre-defined rules to control the order for agents' action,  **Agents**  includes a  _controller_  function that dynamically decides which agent will perform the next action using an LLM by considering the previous actions, the environment, and the target of the current states. This makes multi-agent communication more flexible.
-   **Human-Agent interaction**: In addition to letting language agents communicate with each other in an environment, our framework seamlessly supports human users to play the role of the agent by himself/herself and input his/her own actions, and interact with other language agents in the environment.
-   **Symbolic Control**: Different from existing frameworks for language agents that only use a simple task description to control the entire multi-agent system over the whole task completion process,  **Agents**  allows users to use an  **SOP (Standard Operation Process)**  that defines subgoals/subtasks for the overall task to customize fine-grained workflows for the language agents.

### Links
- Author: [AIWaves Inc.](https:github.com/aiwaves-cn)
- [Paper](https://arxiv.org/pdf/2309.07870.pdf)
- [GitHub Repository](https://github.com/aiwaves-cn/agents)
- [Documentation](https://agents-readthedocsio.readthedocs.io/en/latest/index.html)
- [Tweet](https://twitter.com/wangchunshu/status/1702512370785100133)
</details>

## [AgentVerse](https://github.com/OpenBMB/AgentVerse)
Platform for task-solving & simulation agents
<details>

![Image](https://pbs.twimg.com/card_img/1744672970822615040/m870GGf1?format=jpg&name=medium)

### Category
General purpose, Build your own, Multi-agent

### Description
- Assembles multiple agents to collaboratively accomplish tasks.
- Allows custom environments for observing or interacting with multiple agents.

### Links
- Paper: [AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors](https://arxiv.org/abs/2308.10848)
- [Twitter](https://twitter.com/Agentverse71134)
- [Discord](https://discord.gg/gDAXfjMw)
- [Hugging Face](https://huggingface.co/spaces/AgentVerse/agentVerse)

</details>

## [AI Legion](https://github.com/eumemic/ai-legion)
Multi-agent TS platform, similar to AutoGPT

<details>

![Image](https://res.cloudinary.com/apideck/image/upload/w_1500,f_auto/v1681330426/marketplaces/ckhg56iu1mkpc0b66vj7fsj3o/listings/ai-legion/screenshots/Screenshot_2023-04-12_at_22.13.24_d9kdoj.png)

### Category
Multi-agent, Build-your-own


### Description
- An LLM-powered autonomous agent platform
- A framework for autonomous agents who can work together to accomplish tasks
- Interaction with agents done via console direct messages

### Links
- Author: [eumemic](https://github.com/eumemic)
- [Website](https://gpt3demo.com/apps/ai-legion)
- [GitHub](https://github.com/eumemic/ai-legion)
- [Twitter](https://twitter.com/dysmemic)
</details>

## [Aider](https://github.com/paul-gauthier/aider)
Use command line to edit code in your local repo

<details>


![Image](https://repository-images.githubusercontent.com/638629097/1d3d6251-f8be-4d11-bbb1-4e44b7364b74)

### Category
Coding, GitHub

### Description
- Aider is a command line tool that lets you pair program with GPT-3.5/GPT-4, to edit code stored in your local git repository
- You can start a new project or work with an existing repo. And you can fluidly switch back and forth between the aider chat where you ask GPT to edit the code and your own editor to make changes yourself
- Aider makes sure edits from you and GPT are committed to git with sensible commit messages. Aider is unique in that it works well with pre-existing, larger codebases

### Links  
- [Website](https://aider.chat/)
- Author: [Paul Gauthier](https://github.com/paul-gauthier) (Github)
- [Discord Invite](https://discord.com/invite/Tv2uQnR88V)

</details>

## [AIlice](https://github.com/myshell-ai/AIlice)
Create agents-calling tree to execute your tasks
<details>

![Image](https://github.com/myshell-ai/AIlice/raw/master/AIlice.png)

### Category
General purpose, Personal assistant, Productivity

### Description
- "An Agent in the form of a chatbot independently plans tasks given in natural language and dynamically creates an agents calling tree to execute tasks.
- There is an interaction mechanism between agents to ensure fault tolerance.
- External interaction modules can be automatically built for self-expansion.

### Links  
- [GitHub](https://github.com/myshell-ai/AIlice)

</details>

## [AutoGen](https://github.com/microsoft/autogen)
Multi-agent framework with diversity of agents
<details>

![Image](https://github.com/microsoft/autogen/raw/main/website/static/img/autogen_agentchat.png)

### Category
General purpose, Build your own, Multi-agent

### Description
- A framework for developing LLM (Large Language Model) applications with multiple conversational agents.
- These agents can collaborate to solve tasks and can interact seamlessly with humans.
- It simplifies complex LLM workflows, enhancing automation and optimization.
- It offers a range of working systems across various domains and complexities.
- It improves LLM inference with easy performance tuning and utility features like API unification and caching.
- It supports advanced usage patterns, including error handling, multi-config inference, and context programming.

### Links
- Paper: [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://arxiv.org/pdf/2308.08155.pdf)
- [Discord](https://discord.gg/pAbnFJrkgZ)
- [Twitter thread describing the system](https://twitter.com/pyautogen)


</details>

## [AutoGPT](https://agpt.co/?utm_source=awesome-ai-agents)
Experimental attempt to make GPT4 fully autonomous

<details>

![Image](https://news.agpt.co/wp-content/uploads/2023/04/Logo_-_Auto_GPT-B-800x363.png)

### Category
General purpose

### Description
- An experimental open-source attempt to make GPT-4 fully autonomous, with >140k stars on GitHub
- Chains together LLM "thoughts", to autonomously achieve whatever goal you set
- Internet access for searches and information gathering
- Long-term and short-term memory management
- Can execute many commands such as Google Search, browse websites, write to files, and execute Python files and much more
- GPT-4 instances for text generation
- Access to popular websites and platforms
- File storage and summarization with GPT-3.5
- Extensibility with Plugins
- "A lot like BabyAGI combined with LangChain tools"
- Features added in release 0.4.0
	- File reading
	- Commands customization
	- Enhanced testing

<!--
### Features added in release 0.4.0
- File reading
- Commands customization
- Enhanced testing
-->

### Links
- [Twitter](https://twitter.com/Auto_GPT/?utm_source=awesome-ai-agents)
- [GitHub](https://github.com/Significant-Gravitas/Auto-GPT/?utm_source=awesome-ai-agents)
- [Facebook](https://www.facebook.com/groups/1330282574368178/?utm_source=awesome-ai-agents)
- [Linkedin](https://www.linkedin.com/company/autogpt/?utm_source=awesome-ai-agents)
- [Discord](https://discord.gg/autogpt/?utm_source=awesome-ai-agents)
- Author: [Significant Gravitas](https://twitter.com/SigGravitas/?utm_source=awesome-ai-agents)
</details>



## [Automata](https://github.com/emrgnt-cmplxty/automata)
Generate code based on your project context

<details>


![Image](https://github.com/emrgnt-cmplxty/Automata/assets/68796651/61fe3c33-9b7a-4c1b-9726-a77140476b83)

### Category
Coding

### Description
- Model: GPT 4
- Automata takes your project as a context, receives tasks, and executes the instructions seamlessly.
- Features
	- Automata aims to evolve into a fully autonomous, self-programming Artificial Intelligence system.
	- It's designed for seamless integration with all available agent platforms and LLM providers.
	- Utilizes the novel code search algorithm, SymbolRank, and associated tools to build superior coding intelligence.
	- Modular, fully configurable design with minimal reliance on external dependencies

### Links
- [GitHub](https://github.com/emrgnt-cmplxty/automata)
- [Docs](https://automata.readthedocs.io/en/latest/)
- Author: [Owen Colegrove](https://twitter.com/ocolegro)
<!--

### Features
- Automata aims to evolve into a fully autonomous, self-programming Artificial Intelligence system.
- It's designed for seamless integration with all available agent platforms and LLM providers.
- Utilizes the novel code search algorithm, SymbolRank, and associated tools to build superior coding intelligence.
- Modular, fully configurable design with minimal reliance on external dependencies.

-->

</details>

## [AutoPR](https://github.com/irgolic/AutoPR)
AI-generated pull requests agent that fixes issues

<details>

![Image](https://github.com/irgolic/AutoPR/raw/main/website/static/img/AutoPR_Mark_color.png)

### Category
Coding, GitHub

### Description
- Triggered by adding a label containing AutoPR to an issue, AutoPR will:
	- Plan a fix
	- Write the code
	- Push a branch
	- Open a pull request

### Links
- [Discord](https://discord.com/invite/ykk7Znt3K6)

</details>

## [Autonomous HR Chatbot](https://github.com/stepanogil/autonomous-hr-chatbot)
Agent that answers HR-related queries using tools

<details>

![Image](https://github.com/stepanogil/autonomous-hr-chatbot/raw/main/assets/sample_chat.png)

### Category
HR, Business intelligence, Productivity

### Description
- A prototype enterprise application - an Autonomous HR Assistant powered by GPT-3.5.
- An agent that can answer HR related queries autonomously using the tools it has on hand.
- Powered by GPT-3.5
- Current tools assigned to the agent (with more on the way):
	- Timekeeping Policy
	- Employee Data
	- Calculator

### Links
- Medium: [Creating a (mostly) Autonomous HR Assistant with ChatGPT and LangChain’s Agents and Tools](https://pub.towardsai.net/creating-a-mostly-autonomous-hr-assistant-with-chatgpt-and-langchains-agents-and-tools-1cdda0aa70ef)
- [GitHub](https://github.com/stepanogil/autonomous-hr-chatbot)
- Author: [Stephen Bonifacio](https://twitter.com/Stepanogil)
- [YouTube demo](https://www.youtube.com/watch?v=id7XRcEIBvg&ab_channel=StephenBonifacio)
- [Blog post](https://pub.towardsai.net/creating-a-mostly-autonomous-hr-assistant-with-chatgpt-and-langchains-agents-and-tools-1cdda0aa70ef)
</details>

## [BabyAGI](https://github.com/yoheinakajima/babyagi)
A simple framework for managing tasks using AI
<details>

![Image](https://user-images.githubusercontent.com/21254008/235015461-543a897f-70cc-4b63-941a-2ae3c9172b11.png)

### Category
General purpose

### Description
- A pared-down version of the original [Task-Driven Autonomous Agent](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20)
- Creates tasks based on the result of previous tasks and a predefined objective.
- The script then uses OpenAI's NLP capabilities to create new tasks based on the objective
- Leverages OpenAI's GPT-4, pinecone vector search, and LangChainAI framework
- Default model is OpenAI GPT3-turbo
- The system maintains a task list for managing and prioritizing tasks
- It autonomously creates new tasks based on completed results and reprioritizes the task list accordingly, showcasing the adaptability of AI-powered language models


### Links
- Paper: [Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)
- [Discord](https://discord.com/invite/TMUw26XUcg)
- [Founder's Twitter](https://twitter.com/yoheinakajima)
- [Twitter thread describing the system](https://twitter.com/yoheinakajima/status/1640934493489070080)


</details>


## [BabyBeeAGI](https://yoheinakajima.com/babybeeagi-task-management-and-functionality-expansion-on-top-of-babyagi/)
Task management & functionality BabyAGI expansion

<details>

![Image](https://yoheinakajima.com/wp-content/uploads/2023/04/image.png)

### Category
General purpose, Productivity

### Description
- A more advanced version of the original BabyAGI code
- - Improves upon the original framework, by introducing a more complex task management prompt, allowing for more comprehensive analysis and synthesis of information
- Designed to handle multiple functions within one task management prompt
- Built on top of the GPT-4 architecture, resulting in slower processing speeds and occasional errors
- Provides a framework that can be further built upon and improved, paving the way for more sophisticated AI applications
- One of the significant differences between BabyAGI and BabyBeeAGI is the complexity of the task management prompt

### Links
- [Tweet](https://twitter.com/yoheinakajima/status/1652732735344246784)
- [GitHub](https://github.com/yoheinakajima/babyagi/blob/main/classic/BabyBeeAGI.py)
- [Replit](https://replit.com/@YoheiNakajima/BabyBeeAGI?v=1)
- Author: [@yoheinakajima](https://twitter.com/yoheinakajima) (Twitter)

</details>


## [BabyCatAGI](https://replit.com/@YoheiNakajima/BabyCatAGI)
BabyCatAGI is a mod of BabyBeeAGI
<details>

![Image](https://pbs.twimg.com/media/FwBwoRracAI99iP?format=jpg&name=medium)

### Category
General purpose

### Description
- Just 300 lines of code
- This was built as a d iteration on the original BabyAGI code in a lightweight way. Differences to BabyAGI include the following:
	- Task Creation Agent runs once
	- Execution Agent loops through tasks
	- Task dependencies for pulling relevant results
	- Two tools: search tool and text completion
	- “Mini-agent” as tool
	- Search tool combines search, scrape, chunking, and extraction.
	- Results combined to create summary report


<!--
### How to use
- Fork this into a private Repl
- Add your OpenAI API Key (required) and SerpAPI Key (optional)
- Update the OBJECTIVE variable
- Press "Run" at the top.
-->

### Links
- [Tweet](https://twitter.com/yoheinakajima/status/1657448504112091136)
- [GitHub](https://github.com/yoheinakajima/babyagi/blob/main/classic/BabyCatAGI.py)
- [Replit](https://replit.com/@YoheiNakajima/BabyCatAGI)
- Author: [@yoheinakajima](https://twitter.com/yoheinakajima) (Twitter)

</details>

## [BabyDeerAGI](https://twitter.com/yoheinakajima/status/1666313838868992001)
Mod of BabyAGI with only ~350 lines of code

<details>

![Image](https://pbs.twimg.com/media/Fx_tr0yaUAYP1Q0?format=jpg&name=medium)

### Category
General purpose

### Category
General purpose

### Description
- Features
	- Parallel tasks (making it faster)
	- 3.5-turbo only (GPT-4 not required)
	- User input tool
	- Query rewrite in web search tool
	- Saves results


### Links
- [Tweet](https://twitter.com/yoheinakajima/status/1666313838868992001)
- [GitHub](https://github.com/yoheinakajima/babyagi/blob/main/classic/BabyDeerAGI.py)
- [Replit](https://replit.com/@YoheiNakajima/BabyDeerAGI)
- Author: [@yoheinakajima](https://twitter.com/yoheinakajima) (Twitter)

</details>

## [BabyElfAGI](https://twitter.com/yoheinakajima/status/1678443482866933760)
Mod of BabyDeerAGI, with ~895 lines of code
<details>

![Image](https://pbs.twimg.com/media/F0sHc04aMAEVn3D?format=jpg&name=medium)

### Category
General purpose

### Description
- Features
	- Skills class allows for creation of new skills
	- 'Dynamic task list' example with vector search
	- Beta reflection agent
	- Can read, write, and review its own code


### Links
- [Tweet](https://twitter.com/yoheinakajima/status/1678443482866933760)
- [GitHub](https://github.com/yoheinakajima/babyagi/blob/main/classic/BabyElfAGI/main.py)
- [Replit](https://replit.com/@YoheiNakajima/BabyElfAGI)
- Author: [@yoheinakajima](https://twitter.com/yoheinakajima) (Twitter)

</details>


## [BabyCommandAGI](https://github.com/saten-private/BabyCommandAGI)
Test what happens when you combine CLI and LLM
<details>

![Image](https://github.com/saten-private/BabyCommandAGI/raw/main/docs/Architecture.png)

### Category
General purpose, Coding

### Description
- gent designed to test what happens when you combine CLI and LLM, which are more traditional interfaces than GUI (created by @saten-private)
- An AI agent based on @yoheinakajima's [BabyAGI](https://github.com/yoheinakajima/babyagi) which executes shell commands
- Automatic Programming, Successfully created an app automatically just by providing feedback. The procedure can be found [here](https://twitter.com/saten_work/status/1674855573412810753).
- Automatic Environment Setup, Successfully installed a Flutter environment on Linux in a container, created the Flutter app, and launched it. The procedure can be found [here](https://twitter.com/saten_work/status/1667126272072491009).
- Aside from setting up the environment, it seems to be able to handle a bit of general tasks such as [Generating text, like poems, code, scripts, musical pieces, email, and letters, translating languages](https://anyaitools.com/babycommandagi/?utm_source=SocialAutoPoster&utm_medium=Social&utm_campaign=Twitter)
- There is a risk of breaking the environment. Please run in a virtual environment such as Docker.
- GPT-4 or higher is recommended

### Links
- [Founder's Twitter](https://twitter.com/saten_work)
- [Twitter thread describing the system](https://twitter.com/saten_work/status/1654571194111393793)

</details>


## [BabyFoxAGI](https://github.com/yoheinakajima/babyagi/tree/main/classic/babyfoxagi)
Mod of BabyAGI with a new parallel UI panel


<details>

![Image](https://pbs.twimg.com/media/F2Vpt4EbIAAa326?format=jpg&name=medium)

### Category
General purpose

### Description
- A mod of BabyElfAGI, in a series of mods w the naming of Baby<animal>AGI in alphabetical order
- Self-improving task lists (FOXY method)
   	- By storing a final reflection at the end, and pulling the most relevant reflection to guide future runs, BabyAGI slowly generates better and better tasks lists
- Novel Chat UI w parallel tasks
  	- You can chat w BabyAGI! It has an experimental UI where the chat is separate from the tasks/output panel, allowing you to request multiple tasks in parallel
  	- The Chat UI can use a single skill quickly, or chain multiple skills together using a tasklist
-  New skills
	- 🎨 DALLE skill with prompt assist
 	- 🎶 Music player w Deezer
	- 📊 Airtable search (add your own table/base ID)
	- 🔍 Startup Analyst (example of beefy function call as a skill)
-  It’s own README


### Links
- [Author's Twitter](https://twitter.com/yoheinakajima)
- [Twitter thread describing the system](https://twitter.com/yoheinakajima/status/1697539193768116449)
- [Replit](https://replit.com/@YoheiNakajima)

</details>



## [BambooAI](https://github.com/pgalko/BambooAI)
Data exploration and analysis for non-programmers

<details>

![Image](https://pbs.twimg.com/card_img/1745187734602313730/f-W5kbIU?format=jpg&name=medium)

### Category
Data analysis

### Description
- BambooAI runs in a loop (until user decides to end it).
- Allows mixing of different models with different capabilities, token costs and context windows for different tasks.
- Maintains the memory of previous conversations.
- Builds the prompts dynamically utilising relevant context from Pinecone vector DB.
- Offers a narrative or asks follow up questions if required.
- For codified responses, the task is broken down into a list of steps and a pseudo-code algorithm is built.
- Based on the algorithm, it ises the python code for dataset analysis, modeling or plotting.
- Debugs the code which then executes, auto-corrects if needs to, and displays the output to user.
- Ranks the final answers, and asks user for feedback.
- Builds a vector DB knowledge-base, based on the rank and the user feedback.

### Links
- [GitHub](https://github.com/pgalko/BambooAI)
- [Creators's Twitter](https://twitter.com/pgalko)

</details>


## [BeeBot](https://github.com/AutoPackAI/beebot)
Early-stage project for wide range of tasks

<details>

![Image](https://camo.githubusercontent.com/72231056f7393fa18ee2baa5cedf2688d1fc15478bb6131936e222e5d23ccbb6/68747470733a2f2f6572696b6c702e636f6d2f6d6173636f742e706e67)

### Category
General purpose, Productivity

### Description
- "BeeBot is currently a work in progress and should be treated as an early stage research project. Its focus is not on production usage at this time."

	
### Links
- [GitHub](https://github.com/AutoPackAI/beebot)
- [Tweet](https://twitter.com/Douglas_Schon/status/1681094815021187072?s=20)
</details>


## [Blinky](https://github.com/seahyinghang8/blinky)
An open-source AI debugging agent for VSCode

<details>

![Banner](https://github.com/seahyinghang8/blinky/raw/master/media/banner.png)

### Category
Coding, Debugging

### Description
- Blinky is an open-source AI debugging agent for VSCode that uses LLMs to help identify and fix backend code errors (inspired by SWE-agent).
- Blinky leverages the VSCode API, Language Server Protocol (LSP), and print statement debugging to triangulate and address bugs in real-world backend systems.

	
### Links
- [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=blinky.blinky)
- [Discord](https://discord.gg/d3AUNHDb)
- [GitHub](https://github.com/seahyinghang8/blinky)
</details>


## [Bloop](https://bloop.ai/)
AI code search, works for Rust and Typescript

<details>

![Image](https://bloop.ai/_next/static/media/logo_white.b3bdedc0.svg)

### Category
Coding

### Description
- A GPT-4 powered semantic code search engine that uses an AI agent
- Precise code navigation
- Built on stack graphs and scope queries
- Fast code search and regex matching engine written in Rust
- Allows to find Code on Rust and Typescript
- Allows to stage changes
- The agent searches both your local and remote repositories with natural language, regex and filtered queries
- Bloop can be run via app (easy to download via GitHub)

### Links
- [GitHub](https://github.com/BloopAI/bloop)
- ["Getting started" guide](https://bloop.ai/docs/getting-started)
- [Bloop apps](https://github.com/BloopAI/bloop/releases)

</details>

## [BondAI](https://bondai.dev/)
Code interpreter with CLI & RESTful/WebSocket API

<details>

![Image](https://bondai.dev/assets/images/bondai-logo-9bec7e27b93b804d375221ff8fb6d336.png)

### Category
Coding

### Description
- A highly capable, autonomous AI Agent with an easy to use CLI, RESTful/WebSocket API, Pre-built Docker image and a robust suite of integrated tools.
- Support for all GPT-N, Embeddings and Dall-E OpenAI Models
- Support for Azure OpenAI Services
- Easy to use SDK for integration into any application
- Powerful **Code Interpreter** capabilities
- Powerful data query capabilities via Postgres DB integration
- Pre-built Docker image provides safe execution environment for code generation/execution
- Support for telephony applications (via BlandAI)
- Support for stock trading (via Alpaca Markets)
- Integrates with Gmail and Google Search
- Easy to install `pip install bondai`
- To start the CLI just run `bondai`
- To start the RESTful/WebSocket API just run `bondai --server`

### Links
- [BondAI Homepage/Documentation](https://bondai.dev)
- [Github Repository](https://github.com/krohling/bondai)
- [Docker Image](https://hub.docker.com/r/krohling/bondai)

</details>

## [bumpgen](https://github.com/xeol-io/bumpgen)
AI agent that keeps npm dependencies up-to-date

<details>

![demo](<https://assets-global.website-files.com/65af8f02f12662528cdc93d6/662e6061d42954630a191417_tanstack-ezgif.com-speed%20(1).gif>)

### Category
Coding

### Description
- Put dependency management and upgrades on autopilot
- bumpgen BUMPs an npm package's version up then GENerates the code fixes for breaking changes
- Supports gpt-4-turbo
- Easy install > `npm install -g bumpgen`
- Easy start > `bumpgen @tanstack/react-query 5.28.14`

### Links
- [Repo](https://github.com/xeol-io/bumpgen)
- [Docs](https://docs.xeol.io/bumpgen/home)

</details>

## [Cal.ai](https://cal.ai)
Open-source scheduling assistant built on Cal.com

<details>

![Image](https://3620107743-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FpmUOqZjfGqNkiPmqgnMv%2Fuploads%2F9Qaq1hlaTcqKfrc9k4OG%2Fimage.png?alt=media&token=1ffe8530-19ff-4aea-b020-a99cdc224ce1)

### Category
Productivity

### Description
- Cal.ai can book meetings, summarize your week, and find time with others based on natural language.
- Responds flexibly to unseen tasks eg. "move my second-last meeting to tomorrow morning".
- Uses GPT-4 and LangChain Agent Executor under the hood.
- [GitHub](https://github.com/calcom/cal.com/tree/main/apps/ai)

### Links
- Authors: [Cal.com core team](https://github.com/calcom/cal.com/graphs/contributors), [Dexter Storey](https://github.com/dexterstorey), [Ted Spare](https://github.com/tedspare)

</details>


## [CAMEL](https://github.com/camel-ai/camel)
Architecture for “Mind” Exploration of agents

<details>

![Image](https://raw.githubusercontent.com/camel-ai/camel/master/misc/logo.png)

### Category
General purpose

### Description
- CAMEL is an open-source library designed for the study of autonomous and communicative agents.
1)AI user agent: give instructions to the AI assistant with the goal of completing the task.
2) AI assistant agent: follow AI user’s instructions and respond with solutions to the task
- CAMEL also has an open-source community dedicated to the study of autonomous and communicative agents

### Links
- [Web](https://www.camel-ai.org/)
- [Paper - CAMEL: Communicative Agents for “Mind”
Exploration of Large Scale Language Model Society](https://ghli.org/camel.pdf)
- [Colab demo](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing)
- [GitHub](https://github.com/camel-ai/camel)
- [Hugging face datasets](https://huggingface.co/camel-ai)
- [Slack](https://camel-kwr1314.slack.com/join/shared_invite/zt-1vy8u9lbo-ZQmhIAyWSEfSwLCl2r2eKA#/shared-invite/email)
- [Twitter](https://twitter.com/intent/follow?original_referer=https%3A%2F%2F1508613885-atari-embeds.googleusercontent.com%2F&ref_src=twsrc%5Etfw%7Ctwcamp%5Ebuttonembed%7Ctwterm%5Efollow%7Ctwgr%5ECamelAIOrg&screen_name=CamelAIOrg)
- Authors: Guohao Li∗ Hasan Abed Al Kader Hammoud* Hani Itani* Dmitrii Khizbullin, Bernard Ghanem

</details>

## [ChatArena](https://www.chatarena.org/)
A chat tool for multi agent interaction

<details>

![image](https://github.com/Farama-Foundation/chatarena/raw/main/docs/images/chatarena_architecture.png)

### Category
Design, Build-your-own, SDK for AI apps, Multi-agent

### Description
- ChatArena (or Chat Arena) is a Multi-Agent Language Game Environments for LLMs. The goal is to develop communication and collaboration capabilities of AIs.
ChatArena provides:
- A general framework for building interactive environments for multiple large language models (LLMs). 
- A collection of pre-built or community-created  environments.
- User-friendly interfaces with both Web UI and commandline interfaces.

### Links
- [Web](https://www.chatarena.org/)
- [GitHub](https://github.com/Farama-Foundation/chatarena)
- [X](https://twitter.com/_chatarena)
- [Slack channel](https://chatarena.slack.com/join/shared_invite/zt-1t5fpbiep-CbKucEHdJ5YeDLEpKWxDOg#/shared-invite/email)
  
</details>

## [ChatDev](https://github.com/OpenBMB/ChatDev)
Communicative agents for software development

<details>

![Image](https://github.com/OpenBMB/ChatDev/raw/main/misc/logo1.png)

### Category
Coding, Multi-agent

### Description
- ChatDev is a virtual software company driven by a multitude of intelligent agents assuming different roles such as CEO, CPO, CTO, programmer, reviewer, tester, and art designer, each represented by unique icons.
- These agents collaborate in a structured organizational environment, fulfilling the company's mission to "revolutionize the digital world through programming." They engage in functional seminars focusing on design, coding, testing, and documentation.
- ChatDev aims to provide an accessible, modular, and extensible platform based on large language models, facilitating the study of collective intelligence in a controlled setting.
- The framework allows for extensive customization, empowering users to tailor the software development process, define phases, and establish specific roles within the virtual company.
- ChatDev is committed to open-source principles, encouraging contributions from the community and sharing advancements transparently.

### Links
- [Paper - ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)
- [Local demo](https://github.com/OpenBMB/ChatDev/blob/main/wiki.md#local-demo)
- [GitHub](https://github.com/OpenBMB/ChatDev)

</details>

## [ChemCrow](https://github.com/ur-whitelab/chemcrow-public)
LangChain agent for chemistry-related tasks

<details>

![Image](https://github.com/ur-whitelab/chemcrow-public/raw/main/assets/chemcrow_dark_bold.png)

### Category
Science, Chemistry

### Description
- ChemCrow is an open source package for the accurate solution of reasoning-intensive chemical tasks
- It integrates 13 expert-design tools to augment LLM performance in chemistry and demonstrate effectiveness in automating chemical tasks
- Built with Langchain
- The LLM is provided with a list of tool names, descriptions of their utility, and details about the expected input/output. It is then instructed to answer a user-given prompt using the tools provided when necessary. The instruction suggests the model to follow the ReAct format - Thought, Action, Action Input, Observation. One interesting observation is that while the LLM-based evaluation concluded that GPT-4 and ChemCrow perform nearly equivalently, human evaluations with experts oriented towards the completion and chemical correctness of the solutions showed that ChemCrow outperforms GPT-4 by a large margin. This indicates a potential problem with using LLM to evaluate its own performance on domains that requires deep expertise. The lack of expertise may cause LLMs not knowing its flaws and thus cannot well judge the correctness of task results. (Source: [Weng, Lilian. (Jun 2023). LLM-powered Autonomous Agents". Lil’Log. https://lilianweng.github.io/posts/2023-06-23-agent/.](https://lilianweng.github.io/posts/2023-06-23-agent/))

### Links
- [Paper](https://arxiv.org/abs/2304.05376)
- [GitHub](https://github.com/ur-whitelab/chemcrow-public)
- [HackerNews Discussion](https://news.ycombinator.com/item?id=35607616)

</details>

## [Clippy](https://github.com/ennucore/clippy/)
Agent that can plan, write, debug, and test code

<details>

![Image](https://lev.la/images/clippy.jpg)

### Category
Coding

### Description
- The purpose of Clippy is to elop code for or with the user.
- It can plan, write, debug, and test some projects autonomously.
- For harder tasks, the best way to use it is to look at its work and provide feedback to it.

### Links
- [GitHub](https://github.com/ennucore/clippy/)
- Author: [Lev Chizhov](http://lev.la/) 
