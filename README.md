# Linear Regression DevOps Project

A comprehensive project demonstrating Linear Regression model training, testing, containerization, and CI/CD automation.

## Project Structure

```
lr-devops-project/
├── app.py                          # Main application with Linear Regression model
├── test_app.py                     # Unit tests for the model
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container configuration
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
└── .github/
    └── workflows/
        └── docker-ci.yml          # GitHub Actions CI/CD pipeline
```

## Features

### 1. Linear Regression Model (app.py)
- `LinearRegressionModel` class for training and predictions
- Train, predict, and evaluate methods
- Model persistence (save/load)
- Sample data generation
- Clean, well-documented code

### 2. Comprehensive Testing (test_app.py)
- 11 unit tests covering:
  - Model initialization
  - Training functionality
  - Prediction and evaluation
  - Model persistence
  - Error handling
  - Data shape validation

### 3. Docker Container
- Python 3.10 slim image
- Automatic dependency installation
- Containerized model training and execution

### 4. CI/CD Pipeline (GitHub Actions)
Three-stage automated workflow:
1. **Test Stage**: Runs all unit tests
2. **Build Stage**: Creates Docker image (only if tests pass)
3. **Run Stage**: Executes the container (only if build succeeds)

If tests fail, the pipeline stops and does not proceed to build or run.

## Getting Started

### Local Development

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**:
   ```bash
   python app.py
   ```

4. **Run tests**:
   ```bash
   python -m pytest test_app.py -v
   ```

### Docker Execution

1. **Build Docker image**:
   ```bash
   docker build -t lr-model:latest .
   ```

2. **Run container**:
   ```bash
   docker run --rm lr-model:latest
   ```

## CI/CD Pipeline Workflow

The GitHub Actions pipeline automatically:
- Runs on every push to `main` or `develop` branches
- Runs on all pull requests
- Executes tests first (gating mechanism)
- Only builds Docker if tests pass
- Only runs container if build succeeds

### Pipeline Status Indicators
- ✓ Indicates successful job completion
- ✗ Indicates failed job (stops pipeline)

## Model Details

The model is trained on generated regression data with:
- 100 samples
- 1 feature
- 80% train / 20% test split
- R² evaluation metric

## Dependencies

- **scikit-learn**: Machine learning library
- **numpy**: Numerical computing
- **pytest**: Testing framework

## Testing

Run tests with verbose output:
```bash
pytest test_app.py -v --tb=short
```

Test coverage includes:
- ✓ Model initialization
- ✓ Training process
- ✓ Predictions (before/after training)
- ✓ Evaluation metrics
- ✓ Model serialization
- ✓ Error handling
- ✓ Data shape validation

## GitHub Actions Configuration

The workflow file (`docker-ci.yml`) defines:
- **Trigger**: Push to main/develop, pull requests
- **Jobs**: test → build → run-container
- **Dependencies**: Each job waits for the previous to succeed
- **Artifacts**: Docker image saved for container execution

## Error Handling

- Tests must pass for Docker build to proceed
- Docker build must succeed for container to run
- Clear error messages and success summaries
- Pipeline terminates on first failure

## Troubleshooting

**Tests fail locally**:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest test_app.py -v
```

**Docker build issues**:
```bash
docker build --no-cache -t lr-model:latest .
```

**Check pipeline logs**:
Visit GitHub Actions tab in your repository

## Author Notes

This project demonstrates:
- Software engineering best practices
- Automated testing
- Containerization
- CI/CD automation with GitHub Actions
- Model versioning and persistence
- Professional Python project structure
