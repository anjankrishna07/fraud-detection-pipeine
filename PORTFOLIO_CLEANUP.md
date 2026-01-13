# Portfolio Repository Cleanup Guide

## Files to Remove Before Sharing

This repository contains several files that should **NOT** be committed or shared:

### 🔴 Critical - Remove Before Sharing

1. **`.env` file** - Contains Kaggle API credentials
   ```bash
   rm .env
   ```
   - Use `.env.example` for reference instead
   - Never commit credentials to git

### ⚠️ Generated/Temporary Files (Already in .gitignore)

These are automatically ignored by git but good to clean locally:

1. **Python cache files** - `__pycache__/`, `*.pyc`
2. **Build artifacts** - `*.egg-info/`, `build/`, `dist/`
3. **Virtual environment** - `venv/` (if present)
4. **Data files** - `data/raw/*.csv`, `data/features/*.parquet`
5. **MLflow runs** - `mlruns/` directory
6. **Reports** - `reports/` directory
7. **Log files** - `*.log`, `logs/`

### ✅ Safe to Keep (Professional Structure)

- ✅ Source code (`src/fraud_platform/`)
- ✅ Tests (`tests/`)
- ✅ Documentation (`docs/`, `README.md`)
- ✅ Configuration (`pyproject.toml`, `.env.example`)
- ✅ Docker files (`docker/`, `docker-compose.yml`)
- ✅ Examples (`examples/`)
- ✅ Makefile
- ✅ `.gitignore`

## Quick Cleanup Command

```bash
# Remove sensitive files
rm -f .env

# Clean Python cache (already in .gitignore)
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Clean build artifacts
rm -rf src/*.egg-info/ build/ dist/

# Clean temporary files
rm -f *.log nohup.out .DS_Store

# Clean OS files
find . -name ".DS_Store" -delete 2>/dev/null
```

## Repository Structure for Portfolio

```
fraud-ml-platform/
├── README.md                 # Project documentation
├── pyproject.toml           # Python dependencies
├── Makefile                 # Build commands
├── .env.example             # Environment template (safe to share)
├── .gitignore               # Git ignore rules
├── docker-compose.yml       # Docker orchestration
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.training
├── src/
│   └── fraud_platform/      # Source code (clean, no __pycache__)
├── tests/                   # Test files
├── docs/                    # Documentation
├── examples/                # Example scripts
└── streamlit_app.py         # Web interface
```

## Before Sharing Checklist

- [ ] Remove `.env` file (contains credentials)
- [ ] Verify `.gitignore` is up to date
- [ ] Remove any log files
- [ ] Remove Python cache (`__pycache__/`)
- [ ] Remove build artifacts (`*.egg-info/`)
- [ ] Verify README is comprehensive
- [ ] Check that `.env.example` exists (template only)
- [ ] Ensure sensitive data is not in code
- [ ] Test that repository can be cloned and set up

## GitHub/GitLab Setup

1. Create `.env.example` with placeholder values
2. Ensure `.env` is in `.gitignore`
3. Document setup process in README
4. Add license file (if applicable)
5. Add CONTRIBUTING.md (optional, for open source)

