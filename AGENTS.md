# Agent Guidelines for Digital Arrest Detector

## Build/Lint/Test Commands

### Running the Application
- **Start app**: `streamlit run app.py`
- **Train text model**: `python train_text.py`
- **Evaluate model**: `python evaluate_text_model.py`

### Testing
- Run evaluation script: `python evaluate_text_model.py`
- No dedicated test framework; uses sklearn metrics for evaluation

### Linting
- No explicit linter configured; follow PEP 8 style guide

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports (sklearn, pandas, streamlit, etc.) second
- Local imports last
- Group imports with blank lines between groups

### Naming Conventions
- **Variables/Functions**: snake_case (e.g., `detect_message`, `text_model`)
- **Constants**: UPPER_CASE (e.g., `RISKY_KEYWORDS`, `MODEL_DIR`)
- **Files**: snake_case with .py extension

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Keep under 120 characters when possible
- **Docstrings**: Use triple quotes for function documentation
- **Comments**: Use # for inline comments; keep minimal and descriptive

### Error Handling
- Use try-except blocks for model loading and predictions
- Display errors to user via `st.error()` in Streamlit UI
- Log errors silently in production code

### Types and Type Hints
- No explicit type hints currently used
- Use descriptive variable names to indicate types

### Code Structure
- Group related functionality into sections with comment headers
- Use Path objects for file paths (from pathlib)
- Separate model loading, processing, and UI logic

### Security
- Never log or expose sensitive data
- Validate file uploads and inputs
- Use safe file operations with proper error handling

### Dependencies
- List all dependencies in requirements.txt
- Use specific versions for reproducibility
- Import only what's needed to minimize bundle size