# WellAI Bot AI Agent

## Clone the repository

You can clone the repository using the following command. Make sure to include the `--recurse-submodules` flag to clone the submodules as well.

```sh
git clone --recurse-submodules <repository-url>
```

## Install dependencies

You need to use uv to install dependencies and run the server.

Please see: <https://docs.astral.sh/uv/getting-started/installation/>

After installing uv, you can run the following command to install dependencies:

```sh
uv sync
```

## Start the server

Copy the `.env.example` file to `.env` and set the required environment variables.

To start the gRPC server, run the following command:

```sh
uv run main.py
```

## Architecture

The AI Agent is a gRPC service that receives user messages from the **BotService**, generates AI responses using LangGraph + Groq, and sends replies back through the BotService.

```
WhatsApp User
     |
     v
BotService (stores messages, delivers replies)
     |
     |  GetMessages -> history
     |  Receive(user_message)
     |  <- AI response
     |  Send(ai_reply)
     v
AI Agent (this service)
```

- `AgentService.Receive` — receives a user message from the BotService
- `BotService.GetMessages` — fetches conversation history before generating a response
- `BotService.Send` — forwards the AI reply back to the user

The BotService address is configured via `BOT_SERVICE_ADDR` in `.env`.

## Sending messages to the server

The server exposes an `AgentService` gRPC service on port `50051`. You can talk to it with the included interactive client, the test client, or any gRPC client.

> **Note:** For the full integration (conversation history + reply delivery), a BotService must be running on `BOT_SERVICE_ADDR` (default: `localhost:50052`).

### Interactive chat

An interactive client is included so you can type messages and see AI replies in real time. You need **3 terminals** running:

**Terminal 1: Start the mock BotService**
```sh
uv run mocks/mock_bot_service.py
```

**Terminal 2: Start the AI Agent**
```sh
uv run main.py
```

**Terminal 3: Run the interactive client**
```sh
uv run mocks/interactive_client_with_response.py localhost:50051 localhost:50052
```

Then just type your message and press Enter:

```
> Hello, I want to book an appointment
🤖 AI: Hello! I'd be happy to help you book an appointment...

> I prefer morning slots
🤖 AI: Great, we have several morning slots available...

> quit
Goodbye!
```

You can also pass a custom `user_id` as the third argument:

```sh
uv run mocks/interactive_client_with_response.py localhost:50051 localhost:50052 eliza
```

### Automated test client

To send the built-in test messages (great for quick integration checks):

```sh
uv run mocks/test_client.py localhost:50051
```

### grpcurl (no Python needed)

If you have [grpcurl](https://github.com/fullstorydev/grpcurl) installed, you can call the server directly. First, list the available services:

```sh
grpcurl -plaintext localhost:50051 list
```

Then describe a method:

```sh
grpcurl -plaintext localhost:50051 describe wellai_bot.agent.AgentService
```

Call `Receive` with a single message:

```sh
grpcurl -plaintext -d '{"user_id": "user123", "content": "Hello"}' \
  localhost:50051 wellai_bot.agent.AgentService/Receive
```

## Docker

### Prerequisites

Make sure you have created a `.env` file from the example:

```sh
cp .env.example .env
```

Edit `.env` and add any required secrets (e.g., `GROQ_API_KEY`).

### Build the image

```sh
docker build -t wellai-bot .
```

### Run the container

```sh
docker run --env-file .env --rm -p 50051:50051 wellai-bot
```

The server listens on port `50051` by default.