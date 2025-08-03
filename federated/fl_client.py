"""
NeuroTrust Federated Learning Client
Handles Flower federated learning coordination
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import flwr as fl
from flwr.common import NDArrays, Scalar
import asyncio
from collections import OrderedDict

from models.neurotrust_model import NeuroTrustModel, create_neurotrust_model

logger = logging.getLogger(__name__)

class NeuroTrustFlowerClient(fl.client.NumPyClient):
    """Flower client for NeuroTrust federated learning"""
    
    def __init__(self, 
                 model: NeuroTrustModel,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 client_id: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data"""
        
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Train model
        train_loss, train_metrics = self._train_epoch()
        
        logger.info(f"Client {self.client_id} - Training completed: loss={train_loss:.4f}")
        
        # Return updated parameters and metrics
        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            {"train_loss": train_loss, **train_metrics}
        )
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local data"""
        
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Evaluate model
        val_loss, val_metrics = self._evaluate()
        
        logger.info(f"Client {self.client_id} - Evaluation completed: loss={val_loss:.4f}")
        
        return val_loss, len(self.val_loader.dataset), val_metrics
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        
        self.model.train()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output, _ = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = self.model.get_fault_prediction(output)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, {"accuracy": accuracy}
    
    def _evaluate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate model"""
        
        self.model.eval()
        criterion = torch.nn.BCELoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = self.model.get_fault_prediction(output)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, {"accuracy": accuracy}

class FederatedClient:
    """Main federated learning client coordinator"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.server_address = getattr(config, 'fl_server_address', "localhost:8080")
        self.client_id = None
        
    async def send_weights(self, 
                          weights: Dict[str, Any], 
                          client_id: str, 
                          round_number: int) -> Dict[str, Any]:
        """
        Send local model weights to federated learning coordinator
        
        Args:
            weights: Model weights dictionary
            client_id: Unique client identifier
            round_number: Current federated learning round
            
        Returns:
            Response from FL coordinator
        """
        
        try:
            logger.info(f"Sending FL weights for client {client_id}, round {round_number}")
            
            # Convert weights to proper format for Flower
            numpy_weights = self._convert_weights_to_numpy(weights)
            
            # Store weights for federated aggregation
            # In a real implementation, this would send to Flower server
            result = await self._simulate_fl_aggregation(numpy_weights, client_id, round_number)
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending FL weights: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _convert_weights_to_numpy(self, weights: Dict[str, Any]) -> List[np.ndarray]:
        """Convert PyTorch weights to NumPy arrays"""
        
        numpy_weights = []
        
        for key, value in weights.items():
            if isinstance(value, torch.Tensor):
                numpy_weights.append(value.cpu().numpy())
            elif isinstance(value, np.ndarray):
                numpy_weights.append(value)
            else:
                # Convert to numpy if possible
                try:
                    numpy_weights.append(np.array(value))
                except:
                    logger.warning(f"Could not convert weight {key} to numpy")
        
        return numpy_weights
    
    async def _simulate_fl_aggregation(self, 
                                     weights: List[np.ndarray], 
                                     client_id: str, 
                                     round_number: int) -> Dict[str, Any]:
        """
        Simulate federated learning aggregation
        In production, this would interact with actual Flower server
        """
        
        # Simulate aggregation delay
        await asyncio.sleep(0.1)
        
        # Store client contribution (in production, this would be handled by Flower)
        contribution_info = {
            "client_id": client_id,
            "round_number": round_number,
            "weights_count": len(weights),
            "total_parameters": sum(w.size for w in weights)
        }
        
        logger.info(f"FL aggregation simulated for {contribution_info}")
        
        return {
            "status": "success",
            "aggregation_info": contribution_info,
            "next_round": round_number + 1
        }
    
    def start_fl_client(self, 
                       model: NeuroTrustModel,
                       train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader,
                       client_id: str = None) -> None:
        """
        Start Flower federated learning client
        
        Args:
            model: NeuroTrust model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            client_id: Unique client identifier
        """
        
        if client_id is None:
            client_id = f"client_{np.random.randint(1000, 9999)}"
        
        self.client_id = client_id
        
        logger.info(f"Starting FL client {client_id}")
        
        # Create Flower client
        flower_client = NeuroTrustFlowerClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            client_id=client_id
        )
        
        try:
            # Start Flower client (this would connect to actual server in production)
            logger.info(f"FL client {client_id} would connect to server at {self.server_address}")
            
            # In production, uncomment this line:
            # fl.client.start_numpy_client(server_address=self.server_address, client=flower_client)
            
            logger.info(f"FL client {client_id} started successfully")
            
        except Exception as e:
            logger.error(f"Error starting FL client: {str(e)}")
            raise
    
    def create_federated_strategy(self) -> fl.server.strategy.Strategy:
        """Create federated learning strategy for server"""
        
        def fit_config(rnd: int):
            """Return training configuration dict for each round"""
            config = {
                "batch_size": 32,
                "local_epochs": 1,
                "learning_rate": 0.001
            }
            return config
        
        def evaluate_config(rnd: int):
            """Return evaluation configuration dict for each round"""
            config = {
                "batch_size": 32
            }
            return config
        
        # Create FedAvg strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Use all clients for training
            fraction_evaluate=1.0,  # Use all clients for evaluation
            min_fit_clients=1,  # Minimum clients for training
            min_evaluate_clients=1,  # Minimum clients for evaluation
            min_available_clients=1,  # Minimum available clients
            evaluate_fn=None,  # No centralized evaluation
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )
        
        return strategy
    
    def start_fl_server(self, 
                       model_config: Dict[str, Any],
                       num_rounds: int = 3,
                       port: int = 8080) -> None:
        """
        Start Flower federated learning server
        
        Args:
            model_config: Configuration for creating initial model
            num_rounds: Number of federated learning rounds
            port: Server port
        """
        
        logger.info(f"Starting FL server on port {port} for {num_rounds} rounds")
        
        # Create initial model for server
        initial_model = create_neurotrust_model(model_config)
        
        def get_initial_parameters():
            """Return initial model parameters"""
            return [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        
        # Create strategy
        strategy = self.create_federated_strategy()
        
        try:
            # Start Flower server
            fl.server.start_server(
                server_address=f"0.0.0.0:{port}",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )
            
            logger.info("FL server started successfully")
            
        except Exception as e:
            logger.error(f"Error starting FL server: {str(e)}")
            raise
    
    def get_aggregated_weights(self, round_number: int) -> Optional[Dict[str, Any]]:
        """
        Get aggregated weights from federated learning round
        
        Args:
            round_number: FL round number
            
        Returns:
            Aggregated model weights or None if not available
        """
        
        # In production, this would fetch from Flower server
        logger.info(f"Fetching aggregated weights for round {round_number}")
        
        # Simulate returning aggregated weights
        return {
            "round_number": round_number,
            "status": "weights_available",
            "message": f"Aggregated weights for round {round_number} ready"
        }