# Medical Assistant Chatbot

An AI-powered web application that predicts potential medical conditions based on user-reported symptoms using machine learning. Built with Flask and scikit-learn, this tool provides disease predictions, treatment recommendations, and precautionary measures.

## 🚨 Important Disclaimer

**This application is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.**

## ✨ Features

- **Symptom-based Disease Prediction** - Uses Random Forest classifier for accurate predictions
- **Interactive Web Interface** - User-friendly symptom selection with search functionality
- **Multiple Prediction Results** - Shows top 3 potential diagnoses with confidence scores
- **Comprehensive Information** - Provides disease descriptions, precautions, and treatment suggestions
- **Smart Symptom Matching** - Handles partial matches and typos in symptom input
- **Responsive Design** - Works seamlessly on desktop and mobile devices
- **Model Persistence** - Saves trained models for faster subsequent launches

## 🛠 Technology Stack

- **Backend:** Flask (Python web framework)
- **Machine Learning:** scikit-learn (Random Forest Classifier)
- **Data Processing:** pandas, numpy
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5
- **UI Components:** Select2 for enhanced dropdowns
- **Model Storage:** pickle for model serialization

## 📋 Prerequisites

Before running this application, ensure you have:

- Python 3.7 or higher
- pip (Python package manager)
- Web browser (Chrome, Firefox, Safari, etc.)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd medical-assistant-chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install flask pandas numpy scikit-learn
```

### 4. Prepare Dataset Files

Ensure you have the following CSV files in the `static/` directory:

```
static/
├── dataset.csv              # Main training dataset with symptoms and diseases
├── symptom_Description.csv  # Disease descriptions
└── symptom_precaution.csv   # Precautions for each disease
```

**Dataset Format Requirements:**
- `dataset.csv`: Should contain columns `Symptom_1` through `Symptom_17` and `Disease`
- `symptom_Description.csv`: Should contain `Disease` and `Description` columns
- `symptom_precaution.csv`: Should contain `Disease` and `Precaution_1` through `Precaution_4` columns

## 🏃‍♂️ Running the Application

### Development Mode

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📁 Project Structure

```
medical-assistant-chatbot/
├── app.py                    # Main Flask application
├── templates/
│   └── index.html           # Web interface
├── static/
│   ├── dataset.csv          # Training dataset
│   ├── symptom_Description.csv
│   └── symptom_precaution.csv
├── model.pkl                # Trained ML model (generated)
├── data.pkl                 # Processed data (generated)
├── .gitattributes
└── README.md
```

## 🔧 How It Works

### 1. Data Processing
- Loads symptom-disease datasets from CSV files
- Extracts unique symptoms and creates one-hot encoded feature vectors
- Processes disease descriptions and precautionary measures

### 2. Machine Learning Pipeline
- **Algorithm:** Random Forest Classifier with 100 estimators
- **Features:** Binary symptom presence (1 for present, 0 for absent)
- **Training:** Uses all available data for comprehensive learning
- **Prediction:** Returns top 3 probable diseases with confidence scores

### 3. Symptom Matching
- **Exact Match:** Direct symptom name matching
- **Fuzzy Match:** Partial string matching for typos and variations
- **Smart Selection:** Returns most specific matching symptoms

### 4. Result Generation
- **Primary Diagnosis:** Highest probability disease
- **Confidence Score:** Model prediction probability
- **Alternatives:** Secondary and tertiary diagnoses
- **Recommendations:** Precautions and potential treatments

## 🌐 API Endpoints

### `GET /`
Returns the main web interface

### `GET /get_symptoms`
**Response:** JSON array of all available symptoms
```json
["headache", "fever", "cough", "fatigue", ...]
```

### `POST /predict`
**Request Body:**
```json
{
  "symptoms": ["headache", "fever", "nausea"]
}
```

**Response:**
```json
{
  "disease": "Migraine",
  "confidence": 0.85,
  "alternatives": [
    {"disease": "Migraine", "probability": 0.85},
    {"disease": "Tension Headache", "probability": 0.12}
  ],
  "description": "A migraine is a type of headache...",
  "precautions": ["Rest in a dark room", "Stay hydrated"],
  "medications": ["Sumatriptan", "Ibuprofen"],
  "matched_symptoms": ["headache", "nausea"]
}
```

## 🎯 Usage Guide

### For Users

1. **Access the Application** - Open `http://localhost:5000` in your browser
2. **Select Symptoms** - Choose from dropdown or type to search symptoms
3. **Get Diagnosis** - Click "Check Symptoms" to receive predictions
4. **Review Results** - Examine the primary diagnosis, alternatives, and recommendations
5. **Seek Professional Help** - Consult healthcare providers for proper medical advice

### For Developers

```python
# Initialize the chatbot
chatbot = MedicalChatbot()

# Get all available symptoms
symptoms = chatbot.get_all_symptoms()

# Make a prediction
user_symptoms = ["headache", "fever", "fatigue"]
disease, confidence, alternatives, matched = chatbot.predict_disease(user_symptoms)

# Get additional information
description = chatbot.get_disease_description(disease)
precautions = chatbot.get_precautions(disease)
medications = chatbot.get_medications(disease)
```

## 🔄 Model Training and Updates

### Initial Training
- The model trains automatically on first run using the provided datasets
- Training data is processed and saved as `model.pkl` and `data.pkl`

### Updating the Model
1. **Update Datasets** - Replace CSV files in `static/` directory
2. **Delete Model Files** - Remove `model.pkl` and `data.pkl`
3. **Restart Application** - The model will retrain with new data

### Custom Datasets
To use your own medical datasets:
1. Format CSV files according to the required structure
2. Ensure consistent symptom naming across files
3. Include comprehensive disease information for better predictions

## 🛡️ Privacy and Security

- **No Data Storage** - User symptoms are not stored permanently
- **Local Processing** - All predictions happen locally on the server
- **No Personal Info** - Application doesn't collect personal information
- **Session-based** - No user accounts or persistent data

## 🧪 Testing

### Manual Testing
1. Start the application
2. Try various symptom combinations
3. Verify prediction accuracy against known conditions
4. Test edge cases (no symptoms, unknown symptoms)

### Automated Testing
```bash
# Example unit test structure
python -m pytest tests/ -v
```

## 🚀 Deployment Options

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Cloud Deployment
- **Heroku** - Add `Procfile` with `web: gunicorn app:app`
- **AWS/GCP** - Use their respective Python app services
- **DigitalOcean** - Deploy using App Platform

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/enhancement`)
7. Create a Pull Request

### Contribution Guidelines
- Follow PEP 8 Python style guidelines
- Add docstrings to all functions
- Include tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Legal and Ethical Considerations

- **Educational Purpose Only** - Not intended for actual medical diagnosis
- **No Liability** - Developers assume no responsibility for medical decisions
- **Data Accuracy** - Predictions based on training data quality
- **Professional Consultation** - Always recommend professional medical advice

## 📞 Support and Contact

- **Issues** - Report bugs via GitHub Issues
- **Feature Requests** - Submit enhancement suggestions
- **Documentation** - Check wiki for detailed guides
- **Medical Concerns** - Consult qualified healthcare professionals

## 🔄 Version History

- **v1.0.0** - Initial release with basic symptom prediction
  - Random Forest classifier implementation
  - Web interface with Bootstrap styling
  - Basic symptom matching and disease prediction
  - Precaution and medication recommendations

---

**Remember: This tool is designed to provide general health information and should never replace professional medical consultation. Always seek advice from qualified healthcare providers for medical concerns.**