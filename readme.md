# Project Name

Vanna + Postgres + OpenAI starter

## Introduction

This project is built using the `vanna` package with specific extras for `chromadb`, `openai`, and `postgres`, along with `pandas`. It is designed to be trained on metabase questions, notion documents and various sources

## Installation

To install the necessary dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

This will install the `vanna` package with the specified extras and `pandas`.

## Usage

- Install the requirements.txt
- Place the training material in the root directory
- Create a .env file from the .env.sample file
- Train vanna by running python train_vanna.py
- Ask vanna by running python ask_vanna.py

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.
