# wheter peotry was already installed or not. 
POETRY_INSTALLED := $(shell command -v poetry 2> /dev/null)

# allow ignore present files 
.PHONY: install

# Installation process
install: 
	ifeq ($(POETRY_INSTALLED),)
		@echo "Poetry is not installed. Installing Poetry..."
		curl -sSL https://install.python-poetry.org | python3 -
	else
		@echo "Poetry is already installed."
	endif
		@echo "Installing dependencies using Poetry..."
		poetry install
		@echo "Setup complete!"