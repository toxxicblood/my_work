# Agentic Systems

- With routing and prompt chaining were building sumn called AI workflows but that is different from an AI agent.
- Both are agentic systems but not AI agents.

**Ai workflows**

- This is for when you have pre-defined parts that the data is supposed to take or that the task is supposed to follow.
- This means, even though we have LLM calls, the system is not fully AI managed and thus is just an AI enabled workflow

![image.png](Agentic%20Systems%20223162ab7048808aa602f842cbccb0d8/image.png)

**Ai Agent**

- This is alot more autonomous.
- You give an AI agent a task and it decides which tools to use to complete the task.

![image.png](Agentic%20Systems%20223162ab7048808aa602f842cbccb0d8/image%201.png)

- An example of this is the search mode or deep research mode in LLMs

- Though AI workflows arent as powerful as AI agents, they are very useful in helping automate the mundane tasks in companies.

![image.png](Agentic%20Systems%20223162ab7048808aa602f842cbccb0d8/image%202.png)

**Routing**

![image.png](Agentic%20Systems%20223162ab7048808aa602f842cbccb0d8/image%203.png)

- This applies to when you have input coming in and ur not sure what category it falls into.
- In this case you want an LLM to intelligently decide  what category it falls into and send it down the right path.

![image.png](Agentic%20Systems%20223162ab7048808aa602f842cbccb0d8/image%204.png)

- For example , geting a customer ticket not knowing if it’s a billing issue or a technical issu
- For the above example. the llm categorises the ticket and sends it either to slack or gmail.

There are two types of routing:

***Deterministic:*** 

You know the data thats coming in is well structured and very precictable

![image.png](Agentic%20Systems%20223162ab7048808aa602f842cbccb0d8/image%205.png)

***Non deterministic***

When you get data in unstructure format eg freeform text.