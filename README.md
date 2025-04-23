# Chest Cancer Classification Using Deep Learning

## Project Overview
An end-to-end deep learning solution for automated chest cancer classification from CT scans, implementing a modern MLOps pipeline with PyTorch, FastAPI, and Next.js.

## Features
- Automated classification of chest CT scans into four categories
- Real-time prediction through web interface
- Production-ready MLOps pipeline
- Cloud integration with AWS S3
- Containerized deployment

## Tech Stack
- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: Next.js, Chakra UI
- **MLOps**: DVC, Docker
- **Cloud**: AWS S3
- **CI/CD**: GitHub Actions

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SakxamShrestha/chest-cancer-classifier.git
   cd chest-cancer-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   # Start backend
   uvicorn app.main:app --reload

   # Start frontend (in another terminal)
   cd src/web/frontend
   npm install
   npm run dev
   ```

## Project Structure
```
ml-final-project/
├── data/
│   ├── raw/                  # Raw image data (DVC tracked)
│   └── processed/            # Processed data (DVC tracked)
├── models/                   # Saved model checkpoints (DVC tracked)
├── src/
│   ├── data/                # Data processing scripts
│   ├── models/              # Model architecture
│   ├── training/            # Training scripts
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
├── tests/                   # Unit tests
└── notebooks/              # Jupyter notebooks
```

## Training

To train the model:
```bash
python src/training/train.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License

## Model Architecture
- Base: EfficientNet-B0
- Custom classification head
- Input size: 224x224 pixels
- 4 output classes

## Contributors
- Sakxam Shrestha
- Samyog Karki 
