# Multimodal Evaluation System for Language Models

This repository contains a Python-based backend system built with FastAPI for evaluating and comparing multiple pre-trained language models on various natural language processing (NLP) tasks.

## Features

- **RESTful API**: Implemented using FastAPI to handle requests for evaluating text across multiple language models.
- **Model Integration**: Integrates popular pre-trained models from Hugging Face Transformers for tasks like text classification, named entity recognition (NER), question answering (QA), and text summarization.
- **Caching**: Implements a caching mechanism to store model outputs and improve response times for repeated queries.
- **Benchmarking**: Includes functionality to benchmark models on a provided dataset, calculating metrics such as accuracy, F1 score, and potentially other task-specific metrics.
- **Error Handling**: Proper error handling and input validation for robust API interactions.
- **Logging**: Tracks usage, performance, and errors with integrated logging.
- **Health Endpoint**: Provides a /health endpoint to check the status of the service and basic usage statistics.
- **Optional Features**:
  - Rate limiting to prevent abuse of the API.
  - Authentication system for secure access (can be implemented as needed).

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Poorani-27/Multimodal-Evaluation-and-Comparison-System.git
   cd multimodal-evaluation-system```

2. **Install dependencies**:

```
pip install -r requirements.txt

```

3. **Run the FastAPI server**:

```
uvicorn main:app --reload

```
 - This starts the FastAPI server locally. Access the API at http://localhost:8000.


## API Endpoints

### POST `/evaluate`

- **Description**: Accepts text input and task type (classification, ner, qa, summarization).
- **Returns**: Results from all integrated models for the specified task.

### GET `/benchmark`

- **Description**: Allows benchmarking models on a provided dataset for various NLP tasks.
- **Returns**: Performance metrics (accuracy, F1 score, etc.) for each model.

### GET `/health`

- **Description**: Returns the status of the service and basic usage statistics.


## Usage

### Evaluate Text

Send a POST request to `/evaluate` with JSON payload:

``` {
  "text": "Sample text for evaluation.",
  "task": "classification"
}
```

### GET /benchmark
Description: Allows benchmarking models on a provided dataset for various NLP tasks.
Returns: Performance metrics (accuracy, F1 score, etc.) for each model.
Example Request

``` /benchmark?task=classification&dataset=mydataset.csv ```


### GET /health
Description: Returns the status of the service and basic usage statistics.

