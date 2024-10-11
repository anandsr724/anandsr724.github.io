---
layout: page
title: Personal Chatbot Using Retrieval-Augmented Generation (RAG)
description: A project on RAG
img: assets/img/12.jpg
importance: 1
category: work
related_publications: true
---

This project involved the development of a sophisticated chatbot capable of simulating personalized responses, leveraging Retrieval-Augmented Generation (RAG). The goal was to create a conversational agent that mimics my personal communication style and responds accurately based on context, using a combination of embeddings and vector search.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/project_media/rag_illustration.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    RAG explained with illustraion.
</div>

- **RAG-based System**: The chatbot is built using **RAG**, which combines retrieval of relevant documents or pre-answered questions from a database and natural language generation. This allows the bot to answer queries with both contextually accurate information and fluency, simulating natural conversation.
- **Vector Database**: I created a vectorized database of personal information and pre-answered questions using a document embedding approach. The documents were pre-processed and split into manageable chunks using a **RecursiveCharacterTextSplitter** to improve retrieval efficiency.
- **Hugging Face Embeddings**: For encoding user inputs and database entries, I utilized **OllamaEmbeddings** from Hugging Face. This embedding model translates both user queries and stored responses into vector representations, which are then compared to return the most relevant response. The embedding model helps maintain consistency in response tone and contextual accuracy, making interactions feel more natural.
- **Langchain and Groq Integration**: The system integrates **Langchain** for building retrieval chains and generating responses using **Groq APIs**. By using **Langchain's history-aware retriever**, the chatbot is capable of keeping track of conversation flow, allowing it to provide answers that reflect prior interactions. This feature enables continuity in conversations, especially for multi-turn exchanges, which is crucial for creating a human-like chatbot experience.
- **Conversation Management**: A key aspect of the project is managing chat history. The system stores past interactions in a **ChatMessageHistory** object for each user session, ensuring that previous context is always accessible. This way, the chatbot can refer back to earlier parts of the conversation to improve accuracy and relevance in its responses.
- **Customizable Q&A System**: I designed a custom question-answering (QA) system using **Langchain's prompt templates**. The chatbot generates responses based on a tailored prompt that instructs it to answer questions as if it were me, Anand Sharma, a final-year undergraduate student. This prompt-based customization ensures that the chatbot’s responses reflect my specific knowledge and personality.
- **Flask-based Web Application**: The chatbot was deployed on a **Flask** web application, providing a clean and interactive interface for users. The real-time deployment allows users to type in their questions and receive immediate, human-like responses. The web app structure makes it accessible and easy to interact with on any device.
- **Challenges and Optimizations**: One of the challenges I faced was optimizing the retrieval process for speed without sacrificing accuracy. To address this, I worked on chunking the documents efficiently and setting up the retriever to return the best results based on both query and chat history. The integration of **create_history_aware_retriever** ensured that context wasn't lost even over longer conversations.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/project_media/varta_ss.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A snapshot of the chat bot.
</div>


This project offered valuable experience in:

- Combining state-of-the-art natural language processing techniques like RAG with traditional retrieval systems.
- Embedding-based document retrieval.
- Deploying machine learning applications in real-time environments using **Flask**.
- Managing conversation history and crafting custom prompt chains.

By creating a chatbot that reflects my own personality and knowledge, this project demonstrates the potential of RAG for personal and customizable AI-driven applications.

{% endraw %}
