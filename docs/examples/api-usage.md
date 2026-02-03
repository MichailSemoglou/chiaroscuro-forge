# REST API Examples

This document provides examples of using the Chiaroscuro Forge REST API.

## Installation

The REST API requires FastAPI and uvicorn:

```bash
pip install "fastapi[all]" uvicorn
```

## Starting the Server

```python
from chiaroscuro_forge.api import run_server

# Start the server on port 8000
run_server(port=8000, reload=True)
```

Or from command line:

```bash
uvicorn chiaroscuro_forge.api:app --reload
```

## Authentication

All API endpoints require an API key. You can create a development key:

```python
from chiaroscuro_forge.api import api_key_manager

# Create a new API key
key = api_key_manager.create_key(name="My App", rate_limit=100)
print(f"API Key: {key}")
```

## Processing Images

### Single Image Processing

```python
import requests

# Process an image with gamma correction
with open('input.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'image': f},
        data={'gamma': '1.2', 'application_type': 'photography'},
        headers={'X-API-Key': 'your-api-key'}
    )

job_info = response.json()
job_id = job_info['job_id']
print(f"Job ID: {job_id}")
```

### Checking Job Status

```python
import requests
import time

# Check job status
response = requests.get(
    f'http://localhost:8000/api/v1/jobs/{job_id}',
    headers={'X-API-Key': 'your-api-key'}
)

job_status = response.json()
print(f"Status: {job_status['status']}")
print(f"Progress: {job_status['progress']}")
```

### Complete Workflow

```python
import requests
import time

# Submit processing job
with open('input.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'image': f},
        data={
            'gamma': '1.2',
            'scale_factor': '2.0',
            'application_type': 'medical_imaging'
        },
        headers={'X-API-Key': 'dev-key-12345'}
    )

job_id = response.json()['job_id']

# Poll for completion
while True:
    response = requests.get(
        f'http://localhost:8000/api/v1/jobs/{job_id}',
        headers={'X-API-Key': 'dev-key-12345'}
    )
    
    status = response.json()
    print(f"Progress: {status['progress']*100:.1f}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(1)

if status['status'] == 'completed':
    print(f"Processing complete! Result: {status.get('result_url')}")
else:
    print(f"Processing failed: {status.get('error')}")
```

## API Endpoints

### Health Check

```python
response = requests.get('http://localhost:8000/health')
print(response.json())
# {'status': 'healthy', 'timestamp': '2024-01-15T10:30:00'}
```

### Create API Key (Admin)

```python
response = requests.post(
    'http://localhost:8000/api/v1/keys',
    data={'name': 'Production Key', 'rate_limit': '1000'}
)
print(response.json())
# {'success': True, 'message': '...', 'data': {'api_key': 'sk_...', 'rate_limit': 1000}}
```

## Rate Limiting

Each API key has a rate limit (requests per hour). If you exceed the limit:

```python
response = requests.post(...)
# Status Code: 429 Too Many Requests
# {'detail': 'Rate limit exceeded'}
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Error Handling

```python
try:
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'image': open('input.jpg', 'rb')},
        headers={'X-API-Key': 'dev-key-12345'}
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Invalid API key")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
    elif e.response.status_code == 400:
        print(f"Invalid request: {e.response.json()}")
```

## Security Notes

1. **API Keys**: Store API keys securely (environment variables, secrets manager)
2. **HTTPS**: Use HTTPS in production
3. **Rate Limiting**: Adjust rate limits based on your needs
4. **CORS**: Configure CORS properly for your frontend

## Production Deployment

For production, use a production ASGI server like Gunicorn with Uvicorn workers:

```bash
gunicorn chiaroscuro_forge.api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```
