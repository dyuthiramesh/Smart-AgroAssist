# Smart-AgroAssist

Smart-AgroAssist is an AI-powered application designed to assist farmers and agricultural professionals in diagnosing crop health, providing recommendations, and leveraging knowledge bases for smart agriculture. The project utilizes deep learning models, computer vision, and natural language processing to deliver actionable insights.

## Features
- Crop health diagnosis using image analysis
- Dual domain model for robust predictions
- Integration with knowledge base for recommendations
- Streamlit web interface for user interaction
- Support for external APIs and Groq for advanced NLP

## Project Structure
```
Smart-AgroAssist/
├── app.py                  # Main application entry point (Streamlit UI)
├── model.py                # DualDomainModel definition
├── robust_model_loader.py  # Robust model loading utilities
├── paddy_best_model_classes.json # Class labels for crop models
├── knowledge_base/         # Knowledge base files
├── test_images/            # Sample images for testing
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone <your-repo-url>
cd Smart-AgroAssist
```

### 2. Create and Activate Virtual Environment (Windows)
```powershell
python -m venv venv
Set-ExecutionPolicy RemoteSigned # (Run as Administrator if needed)
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root and add any required API keys or configuration variables. Example:
```
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Run the Application
```powershell
streamlit run app.py
```

### Interact with the Web UI
- Upload crop images for diagnosis
- View predictions and recommendations
- Explore knowledge base articles

## Model Information
- Uses EfficientNet-based models for image classification
- DualDomainModel combines multiple data sources for robust predictions
- Model classes are defined in `paddy_best_model_classes.json`

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes
4. Push to your branch and open a pull request

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please open an issue or contact the maintainer.
