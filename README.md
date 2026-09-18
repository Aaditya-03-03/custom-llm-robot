# Custom LLM Robot

An end-to-end framework integrating Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), tool calling, and safety validation for controlling physical or simulated robotics hardware.

## Project Structure

```text
custom-llm-robot/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── commands.py
│   │   └── health.py
│   ├── llm/
│   │   ├── model.py
│   │   ├── inference.py
│   │   └── prompts.py
│   ├── rag/
│   │   ├── loader.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── vector_db.py
│   ├── memory/
│   │   ├── conversation.py
│   │   └── storage.py
│   ├── tools/
│   │   ├── robot_tools.py
│   │   └── registry.py
│   ├── safety/
│   │   ├── validator.py
│   │   └── limits.py
│   └── schemas/
│       ├── chat.py
│       ├── commands.py
│       └── robot.py
│
├── data/
│   ├── documents/
│   └── vector_db/
│
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

1. Clone or navigate to the repository directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env` template and set up your environment variables.
5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
