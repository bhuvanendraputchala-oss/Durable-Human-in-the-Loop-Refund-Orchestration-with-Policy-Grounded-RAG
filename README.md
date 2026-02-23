# Phase 2 LangGraph Customer Support Triage Agent

A production-grade AI-powered customer support triage system built with LangGraph. The agent automatically classifies support tickets, retrieves relevant company policies via RAG, evaluates policy compliance, proposes refunds with a human approval gate, commits approved refunds, and generates policy-cited customer reply drafts — all with durable PostgreSQL-backed state that survives process restarts.

---

## What the Agent Does

Given a customer support ticket, the graph:

1. **Ingests** the ticket and extracts an order ID from the text via regex (`ORDxxxx`)
2. **Classifies** the issue type using keyword matching (8 types: `refund_request`, `late_delivery`, `missing_item`, `damaged_item`, `duplicate_charge`, `wrong_item`, `defective_product`, `warranty`)
3. **Fetches** the mock order via a `ToolNode` if an order ID is present
4. **Stores** order details into the evidence dict
5. **Proposes** a recommended action based on issue type
6. **Retrieves** relevant policy documents via a multi-stage RAG pipeline (query planning → multi-query retrieval → LLM relevance rescoring → citation extraction)
7. **Evaluates** whether the recommendation complies with retrieved policies (LLM judge)
8. **Routes** to admin review or direct draft based on policy decision
9. **Proposes a refund preview** and **pauses** the graph for human approval (`interrupt_after=["propose_remedy"]`)
10. **Commits the refund** when approved, or records the rejection reason
11. **Drafts a customer reply** with policy citations appended

State is persisted to PostgreSQL at every node. The graph can be fully stopped and resumed from any checkpoint using only the `thread_id`.

---

## Architecture

```
START
  │
  ▼
ingest ──(no ticket)──► END
  │
  ▼
classify_issue
  │
  ├─(order_id found)──► request_fetch_order ──► fetch_order ──► store_order_evidence
  │                                                                       │
  └─(no order_id)──────────────────────────────────────────────────────┘
                                                                         │
                                                                         ▼
                                                               propose_recommendation
                                                                         │
                                                                         ▼
                                                               kb_orchestrator  (RAG)
                                                                         │
                                                                         ▼
                                                               policy_evaluator (LLM)
                                                                         │
                                                                         ▼
                                                                  admin_review
                                                                  │       │
                                                    (refund-eligible)   (other/reject)
                                                                  │       │
                                                                  ▼       ▼
                                                            propose_remedy   draft_reply
                                                         ⚡ INTERRUPT HERE        │
                                                                  │               ▼
                                                          (approved)             END
                                                                  │
                                                                  ▼
                                                            commit_remedy
                                                                  │
                                                                  ▼
                                                             draft_reply
                                                                  │
                                                                  ▼
                                                                 END
```

### RAG Pipeline (kb_orchestrator)

1. **Query Planning** — LLM generates 3 targeted search queries from the issue type and ticket text
2. **Retrieval** — Similarity search (default) or MMR (for `defective_product`, `wrong_item`, `missing_item`); k=4 docs per query
3. **Deduplication** — Duplicate chunks removed across all queries
4. **Relevance Rescoring** — LLM rates each doc 0–10; top 6 returned
5. **Citation Extraction** — source, chunk_id, start_char, and 200-char snippet saved per doc

---

## Repo Structure

```
.
├── app/
│   ├── main.py              # FastAPI endpoints
│   ├── graph.py             # LangGraph graph definition (13 nodes)
│   ├── state.py             # TriageState TypedDict schema
│   ├── tools.py             # fetch_order LangChain tool
│   ├── templates.py         # Customer reply renderer
│   ├── payments.py          # Mock refund preview/commit
│   ├── checkpointer.py      # PostgreSQL checkpointer singleton
│   └── rag/
│       ├── orchestrator.py  # RAG pipeline orchestration
│       ├── vectorstore.py   # ChromaDB vector store setup
│       ├── retriever.py     # Similarity & MMR retrieval
│       └── __init__.py
├── mock_data/
│   ├── orders.json          # 12 mock customer orders (ORD1001–ORD1012)
│   ├── issues.json          # Keyword → issue_type mappings
│   ├── replies.json         # 8 reply templates
│   └── policies/            # 12 markdown policy files (refund, warranty, etc.)
├── tests/
│   ├── test_phase2.py       # 3 integration tests
│   └── conftest.py
├── interactions/
│   └── phase2_demo.json     # 5 example conversation flows
├── scripts/
│   └── kb_index.py          # Knowledge base indexing utility
├── vector_db/               # ChromaDB persistent storage (auto-created)
├── eval_phase1.py           # Phase 1 evaluation script
├── requirements.txt
├── pytest.ini
└── .env                     # Environment variables (do not commit)
```

---

## State Schema

```python
class TriageState(TypedDict, total=False):
    messages: List[AnyMessage]          # LangChain message history
    ticket_text: str                    # Raw customer ticket
    evidence: Dict[str, Any]            # Fetched order details
    order_id: Optional[str]             # Extracted order ID
    issue_type: Optional[str]           # Classified issue type
    recommendation: Optional[str]       # Proposed action
    needs_admin: Optional[bool]         # Requires human approval
    admin_decision: Optional[str]       # "approve" or "reject"
    admin_notes: Optional[str]          # Admin reason/notes
    reply_draft: Optional[str]          # Final customer response
    rag_docs: list                      # Retrieved policy chunks
    citations: list                     # Citation metadata
    policy_decision: str                # LLM policy evaluation output
    policy_allowed: Optional[bool]      # Policy compliance result
    refund_preview: Optional[Dict]      # Pre-execution refund details
    refund_approved: Optional[bool]     # Human approval decision
    refund_reject_reason: Optional[str] # Rejection reason
    remedy_result: Optional[Dict]       # Committed refund result
```

---

## Tech Stack

| Layer | Library |
|---|---|
| API | FastAPI 0.115, Uvicorn 0.30 |
| Graph orchestration | LangGraph 0.2.32 |
| LLM | langchain-openai 0.1.23 (ChatOpenAI) |
| Embeddings | OpenAI (via langchain-openai) |
| Vector DB | ChromaDB 0.5.5 |
| State persistence | langgraph-checkpoint-postgres 3.0.4 |
| PostgreSQL driver | psycopg[binary] 3.3.3 + psycopg-pool |
| Observability | Langfuse 2.60.0 |
| Data validation | Pydantic 2.9.2 |
| Testing | pytest 8.3.4 |

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL running and accessible
- OpenAI API key

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/triage_checkpoints
# Optional — enables Langfuse tracing
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

### 4. Create the PostgreSQL database

```bash
createdb triage_checkpoints
```

The checkpoint tables are created automatically on first startup.

### 5. Index the knowledge base (first run only)

```bash
python scripts/kb_index.py
```

This reads the 12 markdown policy files from `mock_data/policies/` and loads them into ChromaDB at `./vector_db/`.

---

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

API available at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/triage/invoke` | Start a new triage workflow |
| POST | `/triage/resume` | Resume a paused workflow (refund approval) |
| POST | `/triage/admin` | Submit admin decision |
| GET | `/orders/get` | Fetch a single order by order_id |
| GET | `/orders/search` | Search orders by email or query |
| POST | `/classify/issue` | Classify issue type from ticket text |
| POST | `/reply/draft` | Render a customer reply from template |

---

## curl Examples

### Start a triage workflow (refund request)

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I was charged twice for order ORD1003. Please refund the duplicate.",
    "messages": []
  }' | python -m json.tool
```

Response:
```json
{
  "status": "awaiting_admin",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": { "refund_preview": { "preview_id": "prev_a1b2c3d4", "amount": 89.99 } }
}
```

### Resume with admin approval

```bash
curl -s http://127.0.0.1:8000/triage/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "550e8400-e29b-41d4-a716-446655440000",
    "approved": true,
    "reason": "Customer eligible for full refund per policy."
  }' | python -m json.tool
```

Response:
```json
{
  "status": "completed",
  "state": {
    "remedy_result": { "transaction_id": "txn_9f8e7d6c", "status": "committed" },
    "reply_draft": "Hi Alice, we reviewed order ORD1003 and a refund of $89.99 will be processed...\n\nPolicy citations:\n..."
  }
}
```

### Resume with admin rejection

```bash
curl -s http://127.0.0.1:8000/triage/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "550e8400-e29b-41d4-a716-446655440000",
    "approved": false,
    "reason": "Order placed more than 30 days ago."
  }' | python -m json.tool
```

### Empty ticket (graph stops early)

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "", "messages": []}' | python -m json.tool
```

### Ticket without order ID

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "My package arrived broken and I need help.",
    "messages": []
  }' | python -m json.tool
```

---

## Run Tests

```bash
pytest tests/test_phase2.py -v
```

Tests require PostgreSQL to be running (uses the `POSTGRES_URI` from `.env`).

### What the tests cover

| Test | Description |
|---|---|
| `test_persist_interrupt_resume` | Destroys the graph object mid-run, creates a new instance, verifies state is recovered from Postgres and the workflow completes |
| `test_human_approval_gate` | Verifies the graph pauses at `propose_remedy`, that a refund preview is populated, and that approval triggers a committed transaction |
| `test_rag_citations_in_remedy_and_reply` | Verifies RAG citations contain `source`, `chunk_id`, `start_char`, `snippet` and appear in the final reply draft |

---

## Mock Data

**Orders (`mock_data/orders.json`)** — 12 orders, IDs `ORD1001`–`ORD1012`, amounts $29.99–$329.00, statuses: `delivered`, `processing`, `shipped`, `cancelled`.

**Policies (`mock_data/policies/`)** — 12 markdown files covering: refund, warranty, delivery, chargeback, fraud, return, cancellation, dispute resolution, escalation, shipping exceptions, customer data, and pricing policies.

---

## Human-in-the-Loop Flow

The graph uses `interrupt_after=["propose_remedy"]`. When a refund-eligible ticket is processed:

1. **`/triage/invoke`** runs the graph through `propose_remedy` and returns with `"status": "awaiting_admin"` and a `thread_id`
2. The graph state (including the refund preview) is saved to PostgreSQL
3. A human reviews the preview and calls **`/triage/resume`** with the `thread_id` and their decision
4. LangGraph loads the checkpoint, patches the state with the approval/rejection, and continues execution from `commit_remedy`
5. The refund is committed (or rejected with a reason) and a reply draft is generated

This pattern means the process can be stopped between steps 1 and 3 — even across server restarts — and the workflow will resume correctly.

---

## Observability

Graph nodes are instrumented with `langfuse.observe`. Set the `LANGFUSE_*` environment variables before starting the server to enable tracing in your Langfuse project. If the variables are absent, the application runs normally without tracing.

---

## How I Used Claude Code / Cursor

I used Claude Code as an accelerator for implementation and iteration while keeping the architecture intentional. The graph control flow, state schema, and interrupt/resume design were planned before writing code. Claude Code helped with wiring up LangGraph nodes, validating imports, and structuring the RAG pipeline. I used it as a collaborator on implementation details, not as a replacement for reasoning about control flow and state transitions.
