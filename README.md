# Chest Cancer Classifier

An end-to-end deep learning project for chest cancer classification using PyTorch, DVC, and AWS.

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

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/chest-cancer-classifier.git
cd chest-cancer-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure DVC:
```bash
dvc remote add -d s3remote s3://your-bucket-name
```

4. Pull the data:
```bash
dvc pull
```

5. Run the pipeline:
```bash
dvc repro
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
