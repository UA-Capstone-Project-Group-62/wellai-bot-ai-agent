# WellAI Bot AI Agent

## Install dependencies

You need to use uv to install dependencies and run the server.

Please see: <https://docs.astral.sh/uv/getting-started/installation/>

After installing uv, you can run the following command to install dependencies:

```sh
uv sync
```

## Start the server

Copy the `.env.example` file to `.env` and set the `PORT` variable if needed (default is 50051).

To start the gRPC server, run the following command:

```sh
uv run python main.py
```
