from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

app = Flask(__name__)

class MedicalChatbot:
    def __init__(self):
        # Check if model exists, otherwise train it
        if os.path.exists('model.pkl') and os.path.exists('data.pkl'):
            # Load the trained model and data
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('data.pkl', 'rb') as f:
                data = pickle.load(f)
                self.all_symptoms = data['all_symptoms']
                self.model_classes = data['model_classes']
        else:
            # Load data from Kaggle dataset files
            self.dataset = pd.read_csv('static/dataset.csv')
            self.prepare_data()
            self.train_model()
            # Save the model and data for future use
            with open('model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('data.pkl', 'wb') as f:
                data = {
                    'all_symptoms': self.all_symptoms,
                    'model_classes': self.model.classes_
                }
                pickle.dump(data, f)
            self.model_classes = self.model.classes_
            
        # Load other data files
        self.symptom_description = pd.read_csv('static/symptom_Description.csv')
        self.symptom_precaution = pd.read_csv('static/symptom_precaution.csv')
        self.medications_db = self.load_medication_database()
    
    def load_medication_database(self):
        """Create a medication database from symptom_precaution.csv"""
        medications_db = {}
        
        # Map diseases to common medications based on the precautions
        for _, row in self.symptom_precaution.iterrows():
            disease = row['Disease']
            
            # Example medications based on precautions (simplified mapping)
            if 'antibiotic' in str(row).lower():
                meds = ['Amoxicillin', 'Azithromycin', 'Cephalexin']
            elif 'pain' in str(row).lower():
                meds = ['Acetaminophen', 'Ibuprofen', 'Naproxen']
            elif 'allerg' in str(row).lower():
                meds = ['Cetirizine', 'Loratadine', 'Diphenhydramine']
            elif 'rest' in str(row).lower() and 'fluid' in str(row).lower():
                meds = ['Oral rehydration solution', 'Rest', 'Increase fluid intake']
            else:
                # Default medications for common symptoms
                meds = ['Consult with doctor for appropriate medication', 
                        'Follow recommended precautions', 
                        'Symptomatic treatment as advised by healthcare provider']
            
            medications_db[disease] = meds
        
        return medications_db
    
    def prepare_data(self):
        """Process the Kaggle dataset format for model training"""
        # Extract all unique symptoms from the dataset
        all_symptoms = set()
        for i in range(1, 18):  # Dataset has Symptom_1 to Symptom_17
            col = f'Symptom_{i}'
            if col in self.dataset.columns:
                symptoms = self.dataset[col].dropna().unique()
                all_symptoms.update(symptoms)
        
        self.all_symptoms = sorted(list(all_symptoms))
        
        # Create feature matrix (one-hot encoded symptoms)
        X = np.zeros((len(self.dataset), len(self.all_symptoms)))
        for i, row in self.dataset.iterrows():
            for j in range(1, 18):
                col = f'Symptom_{j}'
                if col in row and pd.notna(row[col]) and row[col] in self.all_symptoms:
                    symptom_idx = self.all_symptoms.index(row[col])
                    X[i, symptom_idx] = 1
        
        # Target variable
        y = self.dataset['Disease']
        
        # Use all data for training (no need for test split in production)
        self.X_train, self.y_train = X, y
    
    def train_model(self):
        """Train a Random Forest classifier on the prepared data"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def find_closest_symptom(self, user_symptom):
        """Find the closest matching symptom in our database"""
        user_symptom = user_symptom.lower()
        
        # Direct match
        for symptom in self.all_symptoms:
            if user_symptom == symptom.lower():
                return symptom
        
        # Partial match
        matches = []
        for symptom in self.all_symptoms:
            if user_symptom in symptom.lower() or symptom.lower() in user_symptom:
                matches.append((symptom, len(symptom)))
        
        if matches:
            # Return the shortest matching symptom (often more specific)
            matches.sort(key=lambda x: x[1])
            return matches[0][0]
        
        return None
    
    def predict_disease(self, user_symptoms):
        """Predict disease based on user symptoms"""
        # Create a symptom vector (1 for present symptoms, 0 for absent)
        symptom_vector = np.zeros(len(self.all_symptoms))
        
        matched_symptoms = []
        for symptom in user_symptoms:
            closest = self.find_closest_symptom(symptom)
            if closest:
                matched_symptoms.append(closest)
                symptom_index = self.all_symptoms.index(closest)
                symptom_vector[symptom_index] = 1
        
        # Make prediction
        if sum(symptom_vector) > 0:
            prediction = self.model.predict([symptom_vector])[0]
            probabilities = self.model.predict_proba([symptom_vector])[0]
            probability = np.max(probabilities)
            
            # Get top 3 predictions with probabilities
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_diseases = [(self.model_classes[i], float(probabilities[i])) for i in top_indices]
            
            return prediction, float(probability), top_diseases, matched_symptoms
        else:
            return None, 0, [], []
    
    def get_disease_description(self, disease):
        """Get the description of a disease from the dataset"""
        description = self.symptom_description[
            self.symptom_description['Disease'] == disease]['Description'].values
        
        if len(description) > 0:
            return description[0]
        return "No description available."
    
    def get_precautions(self, disease):
        """Get recommended precautions for a disease"""
        precautions = []
        disease_row = self.symptom_precaution[self.symptom_precaution['Disease'] == disease]
        
        if not disease_row.empty:
            for i in range(1, 5):
                col = f'Precaution_{i}'
                if col in disease_row.columns and pd.notna(disease_row[col].values[0]):
                    precautions.append(disease_row[col].values[0])
        
        return precautions if precautions else ["No specific precautions available."]
    
    def get_medications(self, disease):
        """Get recommended medications for a disease"""
        if disease in self.medications_db:
            return self.medications_db[disease]
        return ["No specific medication information available. Please consult a healthcare professional."]
    
    def get_all_symptoms(self):
        """Return all symptoms for the UI"""
        return self.all_symptoms

# Initialize the chatbot
chatbot = MedicalChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_symptoms')
def get_symptoms():
    symptoms = chatbot.get_all_symptoms()
    return jsonify(symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    if not symptoms:
        return jsonify({
            'error': 'No symptoms provided'
        })
    
    prediction, confidence, top_diseases, matched_symptoms = chatbot.predict_disease(symptoms)
    
    if not prediction:
        return jsonify({
            'error': 'Could not determine a specific condition'
        })
    
    # Format confidence values for JSON
    top_diseases_formatted = []
    for disease, prob in top_diseases:
        top_diseases_formatted.append({
            'disease': disease,
            'probability': prob
        })
    
    description = chatbot.get_disease_description(prediction)
    precautions = chatbot.get_precautions(prediction)
    medications = chatbot.get_medications(prediction)
    
    return jsonify({
        'disease': prediction,
        'confidence': confidence,
        'alternatives': top_diseases_formatted,
        'description': description,
        'precautions': precautions,
        'medications': medications,
        'matched_symptoms': matched_symptoms
    })

if __name__ == '__main__':
    app.run(debug=True)