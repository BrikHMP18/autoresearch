# AutoSales MVP Architecture

## Purpose

Define the smallest end-to-end version of AutoSales that proves the product thesis:

> An agent can run outbound on behalf of an SMB, learn from outcomes, and improve the next campaign.

This MVP is intentionally narrow. It is not a full autonomous sales department. It is a supervised, WhatsApp-first outbound system with one agent, one workflow, one channel, and one measurable feedback loop.

## MVP Definition

The MVP supports:

- One business tenant
- One primary user: founder or sales owner
- One channel: WhatsApp
- One agent
- One outbound goal at a time
- One workflow pattern:
  `find leads -> qualify -> draft WhatsApp flow -> human approve -> send -> handle replies -> book meeting -> measure results -> propose next iteration`

The MVP proves three things:

1. The agent can generate and run a usable outbound workflow.
2. The human can supervise the workflow from a single interface.
3. The system can measure results and suggest the next version of the workflow.

## Non-Goals

The MVP does not include:

- Multi-agent orchestration
- Agent-created agents
- Calls
- Email
- LinkedIn DMs
- Automatic product feedback loops to the engineering team
- Fully autonomous optimization without approval gates
- Full visual workflow editing like n8n
- Broad marketing automation

## Product Surfaces

### CRM View

The CRM View is the operational view for the business user. It should contain:

- Contact and company records
- Lead qualification state
- WhatsApp conversation history
- Pipeline stage
- Meetings booked
- Campaign and workflow performance

### Agents Manager View

The Agents Manager View is the supervision layer. It should contain:

- One active agent card
- Current goal
- Current workflow graph
- Step-by-step execution status
- Logs and agent reasoning summary
- Approval requests
- Metrics for the latest run
- Suggested next iteration

For the MVP, this view is mostly read-only with lightweight intervention:

- Approve or reject a workflow
- Pause or resume execution
- Leave instructions for the next iteration
- Re-run with edits

## User Journey

1. The user connects their WhatsApp business sending account.
2. The user enters business context:
   offer, ICP, geography, value proposition, calendar link, brand voice.
3. The user sets a goal:
   for example, "book meetings with restaurant owners in Lima."
4. The agent creates a workflow proposal:
   target segment, sourcing logic, qualification criteria, message flow, follow-up timing, success metric.
5. The user reviews and approves the workflow.
6. The runtime executes the workflow.
7. Replies are captured, classified, and written into CRM state.
8. Interested leads are escalated to meeting booking.
9. The system computes metrics and stores the run result.
10. The agent proposes the next version of the workflow based on results.

## Core Architecture

The MVP architecture should have five core layers:

1. Frontend
2. API and domain layer
3. Agent layer
4. Workflow runtime
5. Data and analytics layer

### 1. Frontend

Recommended shape:

- Single web app
- Two top-level views:
  CRM View and Agents Manager View
- Real-time updates for workflow step status and conversations

Frontend modules:

- `crm`
- `agents_manager`
- `workflow_graph`
- `conversations`
- `approvals`
- `analytics`

### 2. API and Domain Layer

This layer exposes the system capabilities as stable internal APIs. The agent should never directly manipulate database tables or UI state. It should only use tool-like APIs.

Core domains:

- Leads
- Contacts
- Companies
- Campaigns
- Conversations
- Messages
- Meetings
- Workflow runs
- Metrics

Example internal tools:

- `search_leads`
- `create_contacts`
- `score_contacts`
- `draft_whatsapp_flow`
- `create_workflow_run`
- `request_human_approval`
- `send_whatsapp_message`
- `fetch_inbound_messages`
- `classify_reply`
- `advance_pipeline_stage`
- `book_meeting`
- `compute_campaign_metrics`
- `propose_next_iteration`

### 3. Agent Layer

For the MVP there is one agent with two modes:

- Planner mode
- Operator mode

Planner responsibilities:

- Interpret the business goal
- Build a workflow plan
- Choose qualification rules
- Draft the message flow
- Define success metrics
- Ask for approval before execution

Operator responsibilities:

- Execute the approved workflow through internal APIs
- Observe results
- Trigger follow-ups
- Escalate qualified responses
- Produce a post-run summary and next-step recommendation

Important rule:

The agent should emit a structured workflow spec, not just free-text reasoning. This is what makes visualization, supervision, and iteration possible.

### 4. Workflow Runtime

This is the execution engine for the agent-generated workflow.

Minimal runtime features:

- Step queue
- Retry handling
- Delays and scheduling
- State persistence
- Idempotent step execution
- Step logs
- Human approval gates
- Pause and resume

The runtime executes a workflow graph made of typed nodes.

Minimal node types:

- `source_leads`
- `qualify_leads`
- `draft_messages`
- `await_approval`
- `send_messages`
- `wait_for_replies`
- `classify_replies`
- `book_meetings`
- `update_crm`
- `compute_metrics`
- `suggest_iteration`

### 5. Data and Analytics Layer

The MVP should use one primary transactional database plus one job queue.

Recommended minimum:

- Postgres for transactional data
- Redis or Postgres-backed queue for jobs

Analytics can be computed directly from transactional tables in the MVP. A separate warehouse is not necessary yet.

## Suggested System Components

### Web App

- Hosts CRM View and Agents Manager View
- Displays workflow graph, approvals, logs, and conversations

### API Server

- Owns business logic
- Exposes internal tool endpoints for the agent
- Enforces tenant boundaries, approvals, and permissions

### Agent Service

- Runs the planner and operator loops
- Converts business goals into workflow specs
- Interacts only through internal APIs

### Workflow Worker

- Executes workflow steps asynchronously
- Handles retries, delays, and step transitions

### WhatsApp Gateway

- Sends outbound messages through the chosen provider
- Receives inbound webhook events
- Normalizes provider-specific payloads into internal events

### Scheduler

- Triggers delayed follow-ups
- Polls for stale steps
- Enforces retry windows and rate limits

## Core Data Model

The MVP data model should include these entities:

### Tenant and Identity

- `tenant`
- `user`
- `approval_request`

### CRM

- `company`
- `contact`
- `lead_list`
- `pipeline_stage`
- `meeting`

### Campaign Execution

- `campaign`
- `workflow_template`
- `workflow_run`
- `workflow_step_run`
- `message`
- `conversation`
- `inbound_event`

### Agent State

- `agent`
- `agent_run`
- `agent_note`
- `iteration_recommendation`

### Analytics

- `campaign_metric_snapshot`
- `contact_outcome`

## Minimum Workflow Spec

The workflow spec should be JSON-backed and persistable. It does not need a visual editor in the MVP, but it must be renderable as a graph.

Example shape:

```json
{
  "goal": "Book meetings with restaurant owners in Lima",
  "success_metric": "positive_reply_rate",
  "nodes": [
    { "id": "n1", "type": "source_leads", "config": { "segment": "restaurant owners in Lima", "limit": 50 } },
    { "id": "n2", "type": "qualify_leads", "config": { "rules": ["has_phone", "fits_icp"] } },
    { "id": "n3", "type": "draft_messages", "config": { "style": "direct_and_friendly" } },
    { "id": "n4", "type": "await_approval", "config": {} },
    { "id": "n5", "type": "send_messages", "config": { "daily_limit": 25 } },
    { "id": "n6", "type": "wait_for_replies", "config": { "timeout_hours": 48 } },
    { "id": "n7", "type": "classify_replies", "config": {} },
    { "id": "n8", "type": "book_meetings", "config": {} },
    { "id": "n9", "type": "compute_metrics", "config": {} },
    { "id": "n10", "type": "suggest_iteration", "config": {} }
  ]
}
```

## Minimal Internal API Contract

The first version of the internal API should include:

### Lead and CRM APIs

- `POST /internal/leads/search`
- `POST /internal/contacts/bulk-create`
- `POST /internal/contacts/score`
- `POST /internal/crm/stages/update`
- `POST /internal/meetings/create`

### Workflow APIs

- `POST /internal/workflows/create`
- `POST /internal/workflows/{id}/approve`
- `POST /internal/workflows/{id}/pause`
- `POST /internal/workflows/{id}/resume`
- `GET /internal/workflows/{id}`
- `GET /internal/workflows/{id}/steps`

### Messaging APIs

- `POST /internal/whatsapp/draft`
- `POST /internal/whatsapp/send`
- `POST /internal/whatsapp/webhook`
- `POST /internal/replies/classify`

### Analytics APIs

- `POST /internal/metrics/compute`
- `GET /internal/campaigns/{id}/metrics`
- `POST /internal/iterations/propose`

## WhatsApp-Specific Constraints

WhatsApp is not a generic email replacement. The MVP must treat these as first-class concerns:

- Consent and opt-in status
- Template and policy constraints from the provider
- Number health and messaging limits
- Conversation state and time windows
- Provider webhook reliability

Because of this, each contact should track:

- phone number
- WhatsApp reachability
- consent status
- last outbound message timestamp
- last inbound message timestamp
- conversation state

## Metrics

The MVP should optimize for a small number of metrics:

- messages sent
- delivery success rate
- reply rate
- positive reply rate
- meeting booked rate
- unsubscribe or opt-out rate
- cost per positive reply
- cost per meeting booked

The first optimization loop should target one primary KPI:

- `positive_reply_rate`

Secondary KPI:

- `meetings_booked`

## Safety and Control

The MVP must include guardrails from day one:

- Human approval before first send
- Per-campaign send limits
- Per-day send limits
- Brand voice constraints
- Audit log of all agent actions
- Ability to pause all active workflows
- Clear separation between agent suggestions and executed actions

The agent may recommend changes automatically, but the human must approve rollout of a materially new workflow version.

## Minimal Improvement Loop

This is the AutoResearch-style loop translated to sales:

1. Agent proposes workflow version A
2. Human approves
3. Runtime executes
4. System captures outcomes
5. Metrics are computed
6. Agent analyzes what happened
7. Agent proposes workflow version B
8. Human approves or rejects the next version

At MVP stage, there is no fully autonomous self-modification without review.

## Recommended Tech Shape

A minimal implementation can be built with:

- One web app
- One API service
- One worker service
- One agent service
- Postgres
- Redis or equivalent queue
- One WhatsApp provider integration

If the team wants to keep implementation especially lean, the API service and agent service can start as one deployable unit, as long as the domain boundaries remain clear in code.

## Build Order

### Phase 1: Operational Foundation

- CRM entities
- WhatsApp provider integration
- Conversation storage
- Pipeline stages
- Meeting creation

### Phase 2: Workflow Execution

- Workflow spec format
- Workflow runtime
- Approvals
- Step logs
- Metrics computation

### Phase 3: Agent Layer

- Goal intake
- Workflow generation
- Message drafting
- Iteration recommendation

### Phase 4: Agent Manager View

- Workflow graph visualization
- Run status
- Logs
- Approvals
- KPI summary

## Definition of MVP Done

The MVP is done when a real SMB user can:

1. Connect WhatsApp
2. Give the system a target outbound goal
3. Review and approve an agent-generated workflow
4. Launch the workflow
5. See live step execution and conversations
6. Get meetings booked into the CRM
7. Review a measured recommendation for the next iteration

If those seven actions work reliably for one channel and one ICP, the core thesis is proven.
