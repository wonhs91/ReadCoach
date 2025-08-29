# ReadCoach

A district-wide, privacy-aware reading platform that combines AI-powered recommendations with librarian curation to help students discover books they'll love.

## Features

- **Chat-based AI librarian** for personalized book recommendations
- **Hybrid recommendations** combining semantic search and collaborative filtering
- **Age-appropriate filtering** based on grade level and reading ability
- **Profile management** for personalized experiences
- **Privacy-first design** with FERPA/COPPA compliance considerations

## Quick Start

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Unix/macOS
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key:**
   ```bash
   # Windows
   set OPENAI_API_KEY=sk-your-key-here
   # Unix/macOS
   export OPENAI_API_KEY=sk-your-key-here
   ```

3. **Build the recommendation indexes:**
   ```bash
   python -m app.data_prep --catalog data/catalog.csv --borrows data/borrows.csv
   ```

4. **Start the API server:**
   ```bash
   uvicorn app.api:app --reload
   ```

5. **Test the chat interface:**
   ```bash
   curl -X POST http://127.0.0.1:8000/chat ^
     -H "Content-Type: application/json" ^
     -d "{\"thread_id\":\"demo-s1\",\"message\":\"I liked Smile and New Kid. Can you recommend more graphic novels for 6th grade?\"}"
   ```

## Architecture

- **LangGraph agent** with tool-calling capabilities
- **FAISS vector store** for semantic book search
- **Item-item collaborative filtering** for similar book recommendations
- **FastAPI** REST interface
- **In-memory persistence** for POC (easily upgradeable to production databases)

## Project Structure

```
readcoach_poc/
├─ app/
│  ├─ config.py          # Configuration and paths
│  ├─ data_prep.py       # Data processing and index building
│  ├─ recs.py           # Recommendation engine
│  ├─ tools.py          # LLM-callable tools
│  ├─ agent_graph.py    # LangGraph conversation agent
│  └─ api.py            # FastAPI web interface
├─ data/
│  ├─ catalog.csv       # Book catalog (replace with real data)
│  └─ borrows.csv       # Borrow history (replace with real data)
├─ artifacts/           # Generated indexes (FAISS, similarity)
├─ requirements.txt
└─ README.md
```

## Usage Examples

The AI librarian can help with:
- **Book discovery**: "I want adventure books for 5th graders"
- **Read-alikes**: "I loved Hatchet, what's similar?"
- **Genre exploration**: "Show me graphic novels about friendship"
- **Profile-based recs**: "Save that I'm in 6th grade and like fantasy"

## Next Steps for Production

- Replace sample data with real ILS exports
- Add authentication and rate limiting
- Implement persistent storage (PostgreSQL, Redis)
- Add content safety filters
- Deploy with proper monitoring and logging
- Integrate with school SSO systems

```mermaid
---
graph TD;
	__start__([<p>__start__</p>]):::first
	assistant(assistant)
	tools(tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> assistant;
	assistant -.-> __end__;
	assistant -.-> tools;
	tools --> assistant;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```