"""
NeuroTrust: Federated Neuro-Symbolic Framework for Software Reliability Prediction
Main FastAPI application - UPDATED WITH BALANCED HIGH-PERFORMANCE TRAINER
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
from models.trainer import EnsembleTrainer  # UPDATED: Use ensemble trainer
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
    title="NeuroTrust API - Balanced Performance",
    description="Federated Neuro-Symbolic Framework with Balanced High-Performance Training",
    version="2.0.0"
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
trainer = EnsembleTrainer(config)  # UPDATED: Use ensemble trainer
predictor = ModelPredictor(config)
fl_client = FederatedClient(config)

# Pydantic models for 21 real software metrics features
class PredictionRequest(BaseModel):
    LOC_BLANK: float
    BRANCH_COUNT: float
    LOC_CODE_AND_COMMENT: float
    LOC_COMMENTS: float
    CYCLOMATIC_COMPLEXITY: float
    DESIGN_COMPLEXITY: float
    ESSENTIAL_COMPLEXITY: float
    LOC_EXECUTABLE: float
    HALSTEAD_CONTENT: float
    HALSTEAD_DIFFICULTY: float
    HALSTEAD_EFFORT: float
    HALSTEAD_ERROR_EST: float
    HALSTEAD_LENGTH: float
    HALSTEAD_LEVEL: float
    HALSTEAD_PROG_TIME: float
    HALSTEAD_VOLUME: float
    NUM_OPERANDS: float
    NUM_OPERATORS: float
    NUM_UNIQUE_OPERANDS: float
    NUM_UNIQUE_OPERATORS: float
    LOC_TOTAL: float

class PredictionResponse(BaseModel):
    fault_label: int
    reliability_score: float
    shap_values: Dict[str, float]
    model_confidence: float
    optimal_threshold: float

class FLWeightsRequest(BaseModel):
    weights: Dict[str, Any]
    client_id: str
    round_number: int

class TrainingMetrics(BaseModel):
    epochs_trained: int
    best_val_loss: float
    final_accuracy: float
    final_f1_score: float
    final_precision: float
    final_recall: float
    final_auc: float
    final_balanced_accuracy: float  # ADDED: Balanced accuracy
    optimal_threshold: float
    class_balance_achieved: bool

class DatasetInfo(BaseModel):
    filename: str
    samples: int
    features: int
    target_distribution: Dict[str, int]
    imbalance_ratio: float

class TrainingResponse(BaseModel):
    message: str
    training_metrics: TrainingMetrics
    model_path: str
    dataset_info: DatasetInfo
    performance_summary: Dict[str, str]  # ADDED: Performance summary

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NeuroTrust API - Ensemble Edition",
        "version": "6.0.0 - Ensemble Training",
        "status": "healthy",
        "supported_features": 21,
        "feature_info": "Software metrics with ensemble approach",
        "training_approach": "Ensemble: 5 diverse models to beat Random Forest baseline",
        "enhancements": ["Multiple model architectures", "Diverse training strategies", "Ensemble voting", "Baseline beating"],
        "target_performance": "Beat Random Forest 79.1% accuracy baseline",
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
        "features_supported": 21,
        "dataset_format": "Software metrics (CM1, JM1 compatible)",
        "trainer_type": "DiagnosticAndFixedTrainer",
        "optimization_target": "Root cause analysis + targeted fixes",
        "enhancements": "Comprehensive diagnosis + baseline comparison",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-dataset", response_model=TrainingResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload CSV dataset and train with balanced high-performance approach"""
    try:
        logger.info(f"🏆 ENSEMBLE: Received dataset {file.filename}")
        
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
        
        # Calculate imbalance ratio
        target_counts = processed_data['dataset_info']['target_distribution']
        imbalance_ratio = target_counts[0] / target_counts[1] if target_counts[1] > 0 else 0
        
        logger.info(f"Dataset processed: {len(df)} samples, {len(df.columns)} features")
        logger.info(f"Class imbalance: {imbalance_ratio:.1f}:1")
        logger.info(f"Feature columns: {processed_data['feature_columns']}")
        
        # Train with ensemble approach
        logger.info("🏆 Starting ensemble training to beat Random Forest baseline...")
        model, training_metrics = trainer.train_model(processed_data)
        
        # Save the trained model with all metadata
        model_path = "model/model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_columns': processed_data['feature_columns'],
            'scaler_params': processed_data['scaler_params'],
            'training_metrics': training_metrics,
            'optimal_threshold': training_metrics.get('optimal_threshold', 0.5),
            'model_type': 'Ensemble',
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Ensemble model saved to {model_path}")
        
        # Extract final metrics
        final_metrics = training_metrics['final_metrics']
        
        # Create performance summary
        performance_summary = {}
        
        # Accuracy assessment
        acc = final_metrics['accuracy']
        if acc >= 0.95:
            performance_summary['accuracy'] = f"🏆 Excellent: {acc:.1%}"
        elif acc >= 0.9:
            performance_summary['accuracy'] = f"🎯 Target Achieved: {acc:.1%}"
        elif acc >= 0.85:
            performance_summary['accuracy'] = f"✅ Good: {acc:.1%}"
        else:
            performance_summary['accuracy'] = f"⚠️ Needs Improvement: {acc:.1%}"
        
        # Recall assessment  
        rec = final_metrics['recall']
        if rec >= 0.9:
            performance_summary['recall'] = f"🏆 Excellent: {rec:.1%}"
        elif rec >= 0.8:
            performance_summary['recall'] = f"🎯 Great: {rec:.1%}"
        elif rec >= 0.7:
            performance_summary['recall'] = f"✅ Good: {rec:.1%}"
        else:
            performance_summary['recall'] = f"⚠️ Needs Improvement: {rec:.1%}"
        
        # F1-Score assessment
        f1 = final_metrics['f1_score']
        if f1 >= 0.90:
            performance_summary['f1_score'] = f"🏆 Outstanding: {f1:.1%}"
        elif f1 >= 0.80:
            performance_summary['f1_score'] = f"🎯 Excellent: {f1:.1%}"
        elif f1 >= 0.70:
            performance_summary['f1_score'] = f"✅ Good: {f1:.1%}"
        elif f1 >= 0.60:
            performance_summary['f1_score'] = f"📈 Fair: {f1:.1%}"
        else:
            performance_summary['f1_score'] = f"⚠️ Needs Improvement: {f1:.1%}"
        
        # Overall ultra assessment
        if acc >= 0.90 and rec >= 0.85 and f1 >= 0.80:
            performance_summary['overall'] = "🎉🎉🎉 ULTRA PERFORMANCE ACHIEVED!"
        elif acc >= 0.87 and rec >= 0.80 and f1 >= 0.75:
            performance_summary['overall'] = "🎉 Excellent Performance - Grade A+"
        elif acc >= 0.85 and rec >= 0.75 and f1 >= 0.70:
            performance_summary['overall'] = "🌟 Great Performance - Grade A"
        else:
            performance_summary['overall'] = "📈 Continue Training Needed"
        
        # Create response metrics
        response_metrics = TrainingMetrics(
            epochs_trained=training_metrics['epochs_trained'],
            best_val_loss=training_metrics['best_val_loss'],
            final_accuracy=final_metrics['accuracy'],
            final_f1_score=final_metrics['f1_score'],
            final_precision=final_metrics['precision'],
            final_recall=final_metrics['recall'],
            final_auc=final_metrics['auc'],
            final_balanced_accuracy=final_metrics['balanced_accuracy'],
            optimal_threshold=training_metrics['optimal_threshold'],
            class_balance_achieved=training_metrics.get('class_balance_achieved', True)
        )
        
        # Create dataset info
        target_distribution_str = {str(i): count for i, count in enumerate(target_counts)}
        
        dataset_info = DatasetInfo(
            filename=filename,
            samples=processed_data['dataset_info']['n_samples'],
            features=processed_data['dataset_info']['n_features'],
            target_distribution=target_distribution_str,
            imbalance_ratio=round(imbalance_ratio, 2)
        )
        
        return TrainingResponse(
            message="Dataset uploaded and ensemble model trained successfully",
            training_metrics=response_metrics,
            model_path=model_path,
            dataset_info=dataset_info,
            performance_summary=performance_summary
        )
        
    except Exception as e:
        logger.error(f"Error in ensemble training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ensemble training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the trained balanced model"""
    try:
        logger.info("Received prediction request for smart adaptive model")
        
        # Check if model exists
        if not os.path.exists("model/model.pt"):
            raise HTTPException(
                status_code=400, 
                detail="No trained model found. Please upload a dataset first."
            )
        
        # Make prediction
        result = predictor.predict(request.dict())
        
        # Add optimal threshold to response
        result['optimal_threshold'] = result.get('optimal_threshold', 0.5)
        
        logger.info(f"Smart adaptive prediction: fault_label={result['fault_label']}, "
                   f"confidence={result['model_confidence']:.3f}")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in balanced prediction: {str(e)}")
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
    """Get comprehensive information about the current balanced model"""
    try:
        if not os.path.exists("model/model.pt"):
            return {"message": "No model found"}
        
        checkpoint = torch.load("model/model.pt", map_location='cpu', weights_only=False)
        
        # Get training metrics
        training_metrics = checkpoint.get('training_metrics', {})
        final_metrics = training_metrics.get('final_metrics', {})
        model_config = training_metrics.get('model_config', {})
        
        # Performance evaluation
        performance_eval = {}
        if final_metrics:
            acc = final_metrics.get('accuracy', 0)
            rec = final_metrics.get('recall', 0) 
            f1 = final_metrics.get('f1_score', 0)
            
            performance_eval = {
                'accuracy_status': '🎯 Excellent' if acc >= 0.9 else '✅ Good' if acc >= 0.85 else '⚠️ Needs Work',
                'recall_status': '🎯 Excellent' if rec >= 0.8 else '✅ Good' if rec >= 0.7 else '⚠️ Needs Work', 
                'f1_status': '🎯 Excellent' if f1 >= 0.75 else '✅ Good' if f1 >= 0.65 else '⚠️ Needs Work',
                'overall_grade': 'A+' if (acc >= 0.9 and rec >= 0.8 and f1 >= 0.75) else 
                               'A' if (acc >= 0.85 and rec >= 0.75 and f1 >= 0.7) else
                               'B' if (acc >= 0.8 and rec >= 0.7 and f1 >= 0.6) else 'C'
            }
        
        return {
            "model_exists": True,
            "model_type": checkpoint.get('model_type', 'BalancedHighPerformance'),
            "training_metrics": training_metrics,
            "final_performance": final_metrics,
            "performance_evaluation": performance_eval,
            "feature_columns": checkpoint.get('feature_columns', []),
            "num_features": len(checkpoint.get('feature_columns', [])),
            "model_architecture": model_config,
            "optimal_threshold": training_metrics.get('optimal_threshold', 0.5),
            "class_balanced": training_metrics.get('class_balance_achieved', False),
            "last_trained": checkpoint.get('timestamp', "Unknown"),
            "model_size_mb": round(os.path.getsize("model/model.pt") / (1024*1024), 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {"error": str(e)}

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets with analysis"""
    try:
        data_dir = Path("data")
        datasets = []
        
        for file_path in data_dir.glob("*.csv"):
            stat = file_path.stat()
            
            # Try to analyze dataset
            try:
                df = pd.read_csv(file_path)
                analysis = {
                    "samples": len(df),
                    "features": len(df.columns),
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "has_defect_column": any('defect' in col.lower() for col in df.columns)
                }
            except:
                analysis = {"samples": "Unknown", "features": "Unknown", "size_mb": round(stat.st_size / (1024*1024), 2)}
            
            datasets.append({
                "filename": file_path.name,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "analysis": analysis
            })
        
        return {"datasets": datasets, "total_count": len(datasets)}
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return {"error": str(e)}

@app.get("/features/info")
async def get_feature_info():
    """Get detailed information about supported features"""
    
    feature_descriptions = {
        "LOC_BLANK": "Lines of code that are blank - formatting indicator",
        "BRANCH_COUNT": "Number of branches in the code - control flow complexity",
        "LOC_CODE_AND_COMMENT": "Lines containing both code and comments",
        "LOC_COMMENTS": "Lines containing only comments - documentation level",
        "CYCLOMATIC_COMPLEXITY": "Cyclomatic complexity - key defect predictor",
        "DESIGN_COMPLEXITY": "Design complexity measure - architectural quality",
        "ESSENTIAL_COMPLEXITY": "Essential complexity - structural quality", 
        "LOC_EXECUTABLE": "Executable lines of code - functional size",
        "HALSTEAD_CONTENT": "Halstead content measure - information density",
        "HALSTEAD_DIFFICULTY": "Halstead difficulty - programming effort required",
        "HALSTEAD_EFFORT": "Halstead effort - total development effort",
        "HALSTEAD_ERROR_EST": "Halstead error estimate - predicted defect count",
        "HALSTEAD_LENGTH": "Halstead length - program vocabulary usage",
        "HALSTEAD_LEVEL": "Halstead level - abstraction level",
        "HALSTEAD_PROG_TIME": "Halstead programming time - development duration",
        "HALSTEAD_VOLUME": "Halstead volume - program size in bits",
        "NUM_OPERANDS": "Number of operands - data usage complexity",
        "NUM_OPERATORS": "Number of operators - operational complexity",
        "NUM_UNIQUE_OPERANDS": "Number of unique operands - vocabulary size",
        "NUM_UNIQUE_OPERATORS": "Number of unique operators - operational vocabulary",
        "LOC_TOTAL": "Total lines of code - overall program size"
    }
    
    # Feature importance based on defect prediction research
    feature_importance = {
        "CYCLOMATIC_COMPLEXITY": "🔥 Very High",
        "LOC_EXECUTABLE": "🔥 Very High", 
        "HALSTEAD_DIFFICULTY": "📊 High",
        "HALSTEAD_EFFORT": "📊 High",
        "BRANCH_COUNT": "📊 High",
        "DESIGN_COMPLEXITY": "📊 High",
        "ESSENTIAL_COMPLEXITY": "📊 High",
        "HALSTEAD_VOLUME": "📈 Medium",
        "LOC_TOTAL": "📈 Medium",
        "NUM_OPERATORS": "📈 Medium",
        "NUM_OPERANDS": "📈 Medium",
        "HALSTEAD_ERROR_EST": "📈 Medium",
        "NUM_UNIQUE_OPERATORS": "📉 Low",
        "NUM_UNIQUE_OPERANDS": "📉 Low",
        "HALSTEAD_LENGTH": "📉 Low",
        "HALSTEAD_CONTENT": "📉 Low",
        "LOC_COMMENTS": "📉 Low",
        "HALSTEAD_LEVEL": "📉 Low",
        "LOC_CODE_AND_COMMENT": "📉 Low",
        "LOC_BLANK": "📉 Low",
        "HALSTEAD_PROG_TIME": "📉 Low"
    }
    
    return {
        "total_features": 21,
        "feature_descriptions": feature_descriptions,
        "feature_importance": feature_importance,
        "target_variable": "defect_label (0=No Defect, 1=Defect)",
        "compatible_datasets": ["CM1", "JM1", "KC1", "NASA MDP datasets"],
        "training_approach": "Balanced optimization for all metrics",
        "performance_targets": {
            "accuracy": "≥90%",
            "recall": "≥80%", 
            "f1_score": "≥75%",
            "balanced_accuracy": "≥85%"
        }
    }

@app.get("/model/training-history")
async def get_training_history():
    """Get detailed training history from the balanced model"""
    try:
        if not os.path.exists("model/model.pt"):
            return {"message": "No model found"}
        
        checkpoint = torch.load("model/model.pt", map_location='cpu', weights_only=False)
        training_metrics = checkpoint.get('training_metrics', {})
        
        # Extract training curves
        history = training_metrics.get('training_history', {})
        
        # Find best epoch metrics
        if 'val_accuracy' in history and history['val_accuracy']:
            best_epoch = np.argmax(history['val_accuracy'])
            best_metrics = {
                'best_epoch': best_epoch + 1,
                'best_accuracy': history['val_accuracy'][best_epoch],
                'best_recall': history.get('val_recall', [0])[best_epoch] if best_epoch < len(history.get('val_recall', [])) else 0,
                'best_f1': history.get('val_f1', [0])[best_epoch] if best_epoch < len(history.get('val_f1', [])) else 0
            }
        else:
            best_metrics = {}
        
        return {
            "training_history": history,
            "best_epoch_metrics": best_metrics,
            "model_config": training_metrics.get('model_config', {}),
            "training_info": training_metrics.get('training_info', {}),
            "epochs_trained": training_metrics.get('epochs_trained', 0),
            "optimal_threshold": training_metrics.get('optimal_threshold', 0.5),
            "class_balanced": training_metrics.get('class_balance_achieved', False),
            "feature_count": len(checkpoint.get('feature_columns', []))
        }
        
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}")
        return {"error": str(e)}

@app.get("/model/performance-report")
async def get_performance_report():
    """Get comprehensive performance analysis report"""
    try:
        if not os.path.exists("model/model.pt"):
            return {"message": "No trained model found"}
        
        checkpoint = torch.load("model/model.pt", map_location='cpu', weights_only=False)
        training_metrics = checkpoint.get('training_metrics', {})
        final_metrics = training_metrics.get('final_metrics', {})
        
        if not final_metrics:
            return {"message": "No performance metrics available"}
        
        # Extract key metrics
        accuracy = final_metrics.get('accuracy', 0)
        recall = final_metrics.get('recall', 0)
        precision = final_metrics.get('precision', 0)
        f1_score = final_metrics.get('f1_score', 0)
        balanced_acc = final_metrics.get('balanced_accuracy', 0)
        auc = final_metrics.get('auc', 0)
        
        # Performance grades
        def get_grade(value, excellent=0.9, good=0.8, fair=0.7):
            if value >= excellent:
                return "A+"
            elif value >= good:
                return "A"
            elif value >= fair:
                return "B"
            elif value >= 0.6:
                return "C"
            else:
                return "D"
        
        report = {
            "overall_performance": {
                "accuracy": {"value": f"{accuracy:.1%}", "grade": get_grade(accuracy)},
                "recall": {"value": f"{recall:.1%}", "grade": get_grade(recall, 0.8, 0.7, 0.6)},
                "precision": {"value": f"{precision:.1%}", "grade": get_grade(precision)},
                "f1_score": {"value": f"{f1_score:.1%}", "grade": get_grade(f1_score, 0.85, 0.75, 0.65)},
                "balanced_accuracy": {"value": f"{balanced_acc:.1%}", "grade": get_grade(balanced_acc, 0.85, 0.8, 0.75)},
                "auc": {"value": f"{auc:.3f}", "grade": get_grade(auc, 0.9, 0.8, 0.7)}
            },
            "target_achievement": {
                "accuracy_90_plus": accuracy >= 0.9,
                "recall_80_plus": recall >= 0.8,
                "f1_score_75_plus": f1_score >= 0.75,
                "all_targets_met": accuracy >= 0.9 and recall >= 0.8 and f1_score >= 0.75
            },
            "model_characteristics": {
                "optimal_threshold": training_metrics.get('optimal_threshold', 0.5),
                "epochs_trained": training_metrics.get('epochs_trained', 0),
                "class_balanced_training": training_metrics.get('class_balance_achieved', False),
                "model_type": checkpoint.get('model_type', 'BalancedHighPerformance')
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if accuracy < 0.9:
            report["recommendations"].append("Consider more training epochs or larger model capacity for higher accuracy")
        
        if recall < 0.8:
            report["recommendations"].append("Recall could be improved - consider adjusting class weights or threshold")
        
        if f1_score < 0.75:
            report["recommendations"].append("F1-score indicates room for improvement in precision-recall balance")
        
        if not report["target_achievement"]["all_targets_met"]:
            report["recommendations"].append("Continue training with current balanced approach to achieve all targets")
        else:
            report["recommendations"].append("🎉 Excellent performance! Model ready for production use")
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )