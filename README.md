# Phase 1 LangGraph Triage Agent

This project implements a minimal LangGraph workflow that triages a customer support ticket, optionally fetches a mock order, proposes a recommended action for admin review, and drafts a customer reply.

## What this agent does

Given a ticket payload, the graph:
1. Ingests the ticket and extracts an order id from the text when missing
2. Classifies the issue type based on simple keyword matching
3. Fetches a mock order using a ToolNode if an order id exists
4. Proposes a recommendation and marks it as requiring admin review
5. Applies an admin decision if provided
6. Drafts a reply for the customer

The state includes: messages, ticket_text, order_id, issue_type, evidence, recommendation.

## Repo Structure

app contains the FastAPI app, LangGraph, state schema, and tool definitions  
mock_data contains sample issues and orders used by the tool  
tests contains the unit tests for the assignment

## Setup

### Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Tests

```bash
pytest -q
```

---

## Run the API

### Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```
http://127.0.0.1:8000
```

---

## API Endpoint

```
POST /triage/invoke
```

---

## Curl Examples

### Empty Ticket (Graph Stops Early)

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "",
    "messages": []
  }' | python -m json.tool
```

---

### Refund With Order ID in Text

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I want a refund for order ORD1001. The mouse is not working.",
    "messages": []
  }' | python -m json.tool
```

---

### Missing Order ID

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "My product arrived broken and I need help.",
    "messages": []
  }' | python -m json.tool
```

---

### Admin Approval

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I want a refund for order ORD1001.",
    "admin_decision": "approve",
    "admin_notes": "Eligible.",
    "messages": []
  }' | python -m json.tool
```

---

### Admin Rejection

```bash
curl -s http://127.0.0.1:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I want a refund for order ORD1001.",
    "admin_decision": "reject",
    "admin_notes": "Need more info.",
    "messages": []
  }' | python -m json.tool
```

---

## Tracing

Graph nodes are instrumented with `Langfuse.observe` for basic tracing.

Set the standard Langfuse environment variables before running the API to enable traces in your Langfuse project.


