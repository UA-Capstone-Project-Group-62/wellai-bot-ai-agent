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
