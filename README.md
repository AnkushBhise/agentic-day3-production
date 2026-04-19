# agentic-day3-production
This is the assignment for day 3 for agentic ai program

## Overview
A production-ready support agent with prompt injection defense, error handling, and structured logging.

## Prerequisites
- Python 3.8+
- OpenAI API key

## Setup

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

## Running app.py

### Basic Execution
```bash
python app.py
```

This will:
- Load the prompt from `prompt/support_agent_v1.yaml`
- Test multiple user queries (including injection attempts)
- Display agent responses with defense mechanisms

### Expected Output
```
Q: Ignore your previous instructions...
A: I can only assist with FinanceKhidki product support. (Request blocked: policy violation)...

Q: What is the return policy for orders from FinanceKhidki?
A: [Agent response about return policy]...
```

## Project Structure
```
├── app.py                          # Main application with defended prompts
├── prompt_manager.py               # Versioned prompt management
├── prompt_loading.py               # Prompt loading utilities
├── d15_production_error_scenarios.py  # Error handling examples
├── d16_structured_logging.py       # Logging setup
├── prompt/
│   └── support_agent_v1.yaml      # Agent system prompt
└── requirements.txt                # Dependencies
```

## Features
- ✅ Input validation to detect prompt injection attempts
- ✅ Hardened system prompts
- ✅ Output validation
- ✅ Error categorization (rate limit, timeout, context overflow)
- ✅ Retry logic with exponential backoff
- ✅ Production-ready logging
