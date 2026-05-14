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

## Sending messages to the server

The server exposes an `AgentService` gRPC service on port `50051`. You can talk to it with the included interactive client, the test client, or any gRPC client.

### Interactive chat

An interactive client is included so you can type messages and see AI replies in real time:

```sh
uv run mocks/interactive_client.py localhost:50051
```

Then just type your message and press Enter:

```
> Hello, I want to book an appointment
Bot: Hello! I'd be happy to help you book an appointment...

> I prefer morning slots
Bot: Great, we have several morning slots available...

> quit
Goodbye!
```

You can also pass a custom `user_id` as the second argument:

```sh
uv run mocks/interactive_client.py localhost:50051 eliza
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

## Testing with Mock Servers

A mock gRPC agent server is available in the `mocks/` directory. It runs the **real AI agent logic** from `src/services/agent_service.py` so you can see actual LLM-generated responses without connecting to the full backend stack.

> **Requires:** Your `.env` file must include a valid `GROQ_API_KEY` because the mock server calls the Groq API to generate responses.

### 1. Build the mock server image

```sh
docker build -f mocks/Dockerfile.mock -t mock-agent .
```

### 2. Run the mock server

Keep this running in a terminal so you can watch the logs. Make sure you pass your `.env` file so the container can access `GROQ_API_KEY`:

```sh
docker run --rm -p 50053:50053 --env-file .env mock-agent
```

Or use Docker Compose (it automatically picks up `.env`):

```sh
docker compose -f docker-compose.mock.yaml up --build
```

The mock server will be available on port `50053`.

> **Port already allocated?** If you see `Bind for 0.0.0.0:50053 failed: port is already allocated`, stop the old container first:
>
> ```sh
> docker ps -q --filter publish=50053 | xargs -r docker stop
> ```
>
> Or, if no container is found, kill whatever process is holding the port:
>
> ```sh
> lsof -ti:50053 | xargs kill -9 2>/dev/null
> ```
>
> Then re-run the command above.

### 3. Send test requests and see responses

In a **second terminal**, run the test client:

```sh
uv run mocks/test_client.py localhost:50053
```

**What you will see:**

- **First terminal** (server): logs showing received messages, detected intents, and Groq API calls:
  ```
  INFO:__main__:AI Agent Mock Server running on port 50053
  INFO     | src.services.agent_service:Receive:32 - Received message from user. user_id=user123, content_length=36
  INFO     | src.services.intent_graph:intent_classifier:79 - Detected intent: 'book_app'
  INFO     | src.services.agent_service:Receive:46 - AI replied successfully
  ```

- **Second terminal** (client): the actual AI-generated responses from the agent:
  ```
  Response: success=True, message=Hello, thank you for reaching out to us. We'd be happy to book an appointment for you...

  Response: success=True, message=I see you're interested in morning time slots. We have several available...

  Response: success=True, message=Noted, you'd like to book for next week. Let me check our availability...
  ```