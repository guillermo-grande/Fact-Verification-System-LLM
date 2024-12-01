# Fact Verification System 
This project is a Fact Verification System leveraging Large Language Models (LLMs)
and Retrieval-Augmented Generation (RAG) frameworks. The system validates factual claims
by combining advanced semantic understanding with dynamic, context-aware retrival of 
trusted information. 

## Setup & Dependency Management

To simplify setting up the project in the VM,  we could use `poetry` (for quick dependency management) and `make` to automate the deployment. The following steps allows a quick setup. 

1. Install Poetry (if not installed)
    * The installation link is the [following link](https://python-poetry.org/docs/#installing-with-pipx)

2. Install dependencies using the `Makefile` or poetry directly
```bash
# make version
make install

# poetry version
poetry install
```
3. Deploy and Run the Project using the `Makefile`
```bash
make deploy
```
