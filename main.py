"""
NeuroTrust: Federated Neuro-Symbolic Framework for Software Reliability Prediction
Main FastAPI application - FIXED VERSION
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
import numpy as np
from datetime import datetime

# Import our custom modules
from models.neurotrust_model import NeuroTrustModel
from models.trainer import ModelTrainer
from models.predictor import ModelPredictor
from federated.fl_client import FederatedClient
from utils.data_processor import DataProcessor
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="NeuroTrust API",
    description="Federated Neuro-Symbolic Framework for Software Reliability Prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
data_processor = DataProcessor(config)
trainer = ModelTrainer(config)
predictor = ModelPredictor(config)
fl_client = FederatedClient(config)

# FIXED: Pydantic models for request/response
class PredictionRequest(BaseModel):
    file_complexity: float
    recent_commits: int
    defects: int
    lines_of_code: int
    last_editor_experience: str  # "junior", "mid", "senior"

class PredictionResponse(BaseModel):
    fault_label: int
    reliability_score: float
    shap_values: Dict[str, float]
    model_confidence: float

class FLWeightsRequest(BaseModel):
    weights: Dict[str, Any]
    client_id: str
    round_number: int

# FIXED: Simplified training metrics model
class TrainingMetrics(BaseModel):
    epochs_trained: int
    best_val_loss: float
    final_accuracy: float
    final_f1_score: float
    final_precision: float
    final_recall: float
    final_auc: float

class DatasetInfo(BaseModel):
    filename: str
    samples: int
    features: int
    target_distribution: Dict[str, int]

class TrainingResponse(BaseModel):
    message: str
    training_metrics: TrainingMetrics
    model_path: str
    dataset_info: DatasetInfo

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NeuroTrust API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_exists = os.path.exists("model/model.pt")
    return {
        "api_status": "healthy",
        "model_loaded": model_exists,
        "data_directory": os.path.exists("data"),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-dataset", response_model=TrainingResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload CSV dataset and train the NeuroTrust model"""
    try:
        logger.info(f"Received dataset upload: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}_{file.filename}"
        filepath = f"data/{filename}"
        
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"Dataset saved to {filepath}")
        
        # Process the data
        df = pd.read_csv(filepath)
        processed_data = data_processor.process_dataset(df)
        
        logger.info(f"Dataset processed: {len(df)} samples, {len(df.columns)} features")
        
        # Train the model
        model, training_metrics = trainer.train_model(processed_data)
        
        # Save the trained model
        model_path = "model/model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_columns': processed_data['feature_columns'],
            'scaler_params': processed_data['scaler_params'],
            'training_metrics': training_metrics,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        # FIXED: Extract metrics properly for response
        final_metrics = training_metrics['final_metrics']
        
        # Create simplified training metrics
        simplified_metrics = TrainingMetrics(
            epochs_trained=training_metrics['epochs_trained'],
            best_val_loss=training_metrics['best_val_loss'],
            final_accuracy=final_metrics.get('accuracy', 0.0),
            final_f1_score=final_metrics.get('f1_score', 0.0),
            final_precision=final_metrics.get('precision', 0.0),
            final_recall=final_metrics.get('recall', 0.0),
            final_auc=final_metrics.get('auc', 0.0)
        )
        
        # Create dataset info - FIXED: Convert keys to strings
        target_counts = df.get('defects', pd.Series()).value_counts().to_dict()
        target_distribution_str = {str(k): v for k, v in target_counts.items()}
        
        dataset_info = DatasetInfo(
            filename=filename,
            samples=len(df),
            features=len(df.columns),
            target_distribution=target_distribution_str
        )
        
        return TrainingResponse(
            message="Dataset uploaded and model trained successfully",
            training_metrics=simplified_metrics,
            model_path=model_path,
            dataset_info=dataset_info
        )
        
    except Exception as e:
        logger.error(f"Error in upload_dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the trained NeuroTrust model"""
    try:
        logger.info("Received prediction request")
        
        # Check if model exists
        if not os.path.exists("model/model.pt"):
            raise HTTPException(
                status_code=400, 
                detail="No trained model found. Please upload a dataset first."
            )
        
        # Make prediction
        result = predictor.predict(request.dict())
        
        logger.info(f"Prediction completed: fault_label={result['fault_label']}")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/fl/upload-weights")
async def upload_fl_weights(request: FLWeightsRequest):
    """Upload local model weights for federated learning aggregation"""
    try:
        logger.info(f"Received FL weights from client {request.client_id}")
        
        # Send weights to Flower coordinator
        result = await fl_client.send_weights(
            weights=request.weights,
            client_id=request.client_id,
            round_number=request.round_number
        )
        
        logger.info(f"FL weights processed for round {request.round_number}")
        
        return {
            "message": "Weights uploaded successfully",
            "client_id": request.client_id,
            "round_number": request.round_number,
            "status": result.get("status", "success")
        }
        
    except Exception as e:
        logger.error(f"Error in upload_fl_weights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FL upload failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    try:
        if not os.path.exists("model/model.pt"):
            return {"message": "No model found"}
        
        checkpoint = torch.load("model/model.pt", map_location='cpu')
        
        return {
            "model_exists": True,
            "training_metrics": checkpoint.get('training_metrics', {}),
            "feature_columns": checkpoint.get('feature_columns', []),
            "last_trained": checkpoint.get('timestamp', "Unknown"),
            "model_size": os.path.getsize("model/model.pt")
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {"error": str(e)}

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    try:
        data_dir = Path("data")
        datasets = []
        
        for file_path in data_dir.glob("*.csv"):
            stat = file_path.stat()
            datasets.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return {"error": str(e)}

# BONUS: Endpoint to get detailed training history
@app.get("/model/training-history")
async def get_training_history():
    """Get detailed training history from the saved model"""
    try:
        if not os.path.exists("model/model.pt"):
            return {"message": "No model found"}
        
        checkpoint = torch.load("model/model.pt", map_location='cpu')
        training_metrics = checkpoint.get('training_metrics', {})
        
        return {
            "training_history": training_metrics.get('training_history', {}),
            "model_config": training_metrics.get('model_config', {}),
            "epochs_trained": training_metrics.get('epochs_trained', 0),
            "best_val_loss": training_metrics.get('best_val_loss', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )