# IOFT Humanoid Robot Knowledge Base

The IOFT Humanoid Robot is an intelligent bipedal/locomotion humanoid robot designed for interaction and physical navigation.

## Architecture Overview
- **Human-Robot Interaction Layer**: Meta Quest VR headset running a custom Unity application for spatial awareness and human-robot communication.
- **Microcontroller Hardware**: ESP32-based controllers manage motor drivers, sensor acquisition, and real-time safety watchdogs.
- **AI Brain & Intelligence Server**: Local Python FastAPI server running modular microservices.
- **Local Large Language Model**: Local Qwen 2.5 3B model (fine-tuned `ioft-qwen25-3b-v1` via QLoRA) running through Ollama for deterministic tool planning and conversational assistance.
- **Conversation Memory & RAG**: MongoDB for multi-turn conversational session history and ChromaDB for retrieval-augmented documentation grounding.
- **Safety Architecture**: Multi-stage fail-closed execution pipeline ensuring untrusted model outputs never bypass validation, authority gates, or telemetry state checks.
