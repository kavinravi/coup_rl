import os
import json
import torch
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np


class CheckpointManager:
    """manages training checkpoints with automatic cleanup and best model tracking"""
    
    def __init__(self, checkpoint_dir: str, config: Dict[str, Any]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.checkpoints_config = config.get('checkpoints', {})
        
        # checkpoint tracking
        self.best_checkpoints = []  # list of (metric_value, checkpoint_path)
        self.latest_checkpoints = []  # list of (timestep, checkpoint_path)
        self.checkpoint_metadata = {}  # timestep -> metadata dict
        
        # configuration
        self.keep_best = self.checkpoints_config.get('keep_best', 5)
        self.keep_latest = self.checkpoints_config.get('keep_latest', 3)
        self.save_optimizer = self.checkpoints_config.get('save_optimizer', True)
        self.save_training_state = self.checkpoints_config.get('save_training_state', True)
        
        # metadata file
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self._load_metadata()
        
    def _load_metadata(self):
        """load existing checkpoint metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_metadata = data.get('checkpoints', {})
                    self.best_checkpoints = data.get('best_checkpoints', [])
                    self.latest_checkpoints = data.get('latest_checkpoints', [])
            except (json.JSONDecodeError, KeyError):
                self.checkpoint_metadata = {}
                self.best_checkpoints = []
                self.latest_checkpoints = []
    
    def _save_metadata(self):
        """save checkpoint metadata"""
        data = {
            'checkpoints': self.checkpoint_metadata,
            'best_checkpoints': self.best_checkpoints,
            'latest_checkpoints': self.latest_checkpoints,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_checkpoint(self, timestep: int, agent, trainer, metrics: Dict[str, Any],
                       is_best: bool = False, checkpoint_name: Optional[str] = None) -> str:
        """save training checkpoint"""
        
        # create checkpoint name
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{timestep}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        # prepare checkpoint data
        checkpoint_data = {
            'timestep': timestep,
            'agent_state_dict': agent.network.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'save_time': datetime.now().isoformat()
        }
        
        # optionally save optimizer state
        if self.save_optimizer and hasattr(trainer, 'optimizer'):
            checkpoint_data['optimizer_state_dict'] = trainer.optimizer.state_dict()
        
        # optionally save training state
        if self.save_training_state and hasattr(trainer, 'stats'):
            checkpoint_data['training_stats'] = trainer.stats
        
        # save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # update metadata
        self.checkpoint_metadata[str(timestep)] = {
            'path': str(checkpoint_path),
            'timestep': timestep,
            'metrics': metrics,
            'is_best': is_best,
            'save_time': datetime.now().isoformat()
        }
        
        # update tracking lists
        if is_best:
            self._update_best_checkpoints(metrics.get('elo_rating', 0), str(checkpoint_path))
        
        self._update_latest_checkpoints(timestep, str(checkpoint_path))
        
        # cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # save metadata
        self._save_metadata()
        
        return str(checkpoint_path)
    
    def _update_best_checkpoints(self, metric_value: float, checkpoint_path: str):
        """update list of best checkpoints"""
        # add new checkpoint
        self.best_checkpoints.append((metric_value, checkpoint_path))
        
        # sort by metric value (descending)
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # keep only top k
        if len(self.best_checkpoints) > self.keep_best:
            # remove excess checkpoints
            removed = self.best_checkpoints[self.keep_best:]
            self.best_checkpoints = self.best_checkpoints[:self.keep_best]
            
            # delete checkpoint files
            for _, path in removed:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
    
    def _update_latest_checkpoints(self, timestep: int, checkpoint_path: str):
        """update list of latest checkpoints"""
        # add new checkpoint
        self.latest_checkpoints.append((timestep, checkpoint_path))
        
        # sort by timestep (descending)
        self.latest_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # keep only latest k
        if len(self.latest_checkpoints) > self.keep_latest:
            # remove excess checkpoints
            removed = self.latest_checkpoints[self.keep_latest:]
            self.latest_checkpoints = self.latest_checkpoints[:self.keep_latest]
            
            # delete checkpoint files (if not in best)
            best_paths = set(path for _, path in self.best_checkpoints)
            for _, path in removed:
                if path not in best_paths:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError:
                        pass
    
    def _cleanup_checkpoints(self):
        """cleanup old checkpoints based on policy"""
        # get all tracked checkpoint paths
        tracked_paths = set()
        tracked_paths.update(path for _, path in self.best_checkpoints)
        tracked_paths.update(path for _, path in self.latest_checkpoints)
        
        # find all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        
        # remove untracked checkpoints
        for checkpoint_file in checkpoint_files:
            if str(checkpoint_file) not in tracked_paths:
                try:
                    checkpoint_file.unlink()
                except OSError:
                    pass
    
    def load_checkpoint(self, checkpoint_path: str, agent, trainer=None) -> Dict[str, Any]:
        """load checkpoint and restore state"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=agent.device)
        
        # restore agent state
        agent.network.load_state_dict(checkpoint_data['agent_state_dict'])
        
        # restore optimizer state
        if trainer and self.save_optimizer and 'optimizer_state_dict' in checkpoint_data:
            trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # restore training state
        if trainer and self.save_training_state and 'training_stats' in checkpoint_data:
            trainer.stats = checkpoint_data['training_stats']
        
        return checkpoint_data
    
    def load_best_checkpoint(self, agent, trainer=None) -> Optional[Dict[str, Any]]:
        """load the best checkpoint"""
        if not self.best_checkpoints:
            return None
        
        # get best checkpoint path
        best_path = self.best_checkpoints[0][1]
        
        return self.load_checkpoint(best_path, agent, trainer)
    
    def load_latest_checkpoint(self, agent, trainer=None) -> Optional[Dict[str, Any]]:
        """load the latest checkpoint"""
        if not self.latest_checkpoints:
            return None
        
        # get latest checkpoint path
        latest_path = self.latest_checkpoints[0][1]
        
        return self.load_checkpoint(latest_path, agent, trainer)
    
    def get_checkpoint_info(self, timestep: int) -> Optional[Dict[str, Any]]:
        """get information about a specific checkpoint"""
        return self.checkpoint_metadata.get(str(timestep))
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """list all available checkpoints"""
        checkpoints = []
        
        for timestep, metadata in self.checkpoint_metadata.items():
            checkpoints.append({
                'timestep': int(timestep),
                'path': metadata['path'],
                'metrics': metadata['metrics'],
                'is_best': metadata['is_best'],
                'save_time': metadata['save_time']
            })
        
        # sort by timestep
        checkpoints.sort(key=lambda x: x['timestep'])
        
        return checkpoints
    
    def get_best_checkpoints(self) -> List[Tuple[float, str]]:
        """get list of best checkpoints"""
        return self.best_checkpoints.copy()
    
    def get_latest_checkpoints(self) -> List[Tuple[int, str]]:
        """get list of latest checkpoints"""
        return self.latest_checkpoints.copy()
    
    def cleanup_all_checkpoints(self):
        """remove all checkpoints and metadata"""
        # remove all checkpoint files
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint_file.unlink()
            except OSError:
                pass
        
        # remove metadata
        if self.metadata_file.exists():
            try:
                self.metadata_file.unlink()
            except OSError:
                pass
        
        # clear internal state
        self.checkpoint_metadata = {}
        self.best_checkpoints = []
        self.latest_checkpoints = []
    
    def create_checkpoint_summary(self) -> Dict[str, Any]:
        """create summary of all checkpoints"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return {'total_checkpoints': 0}
        
        # collect metrics
        elo_ratings = [c['metrics'].get('elo_rating', 0) for c in checkpoints]
        win_rates = []
        
        for checkpoint in checkpoints:
            win_rate_sum = 0
            win_rate_count = 0
            for key, value in checkpoint['metrics'].items():
                if key.startswith('win_rate_'):
                    win_rate_sum += value
                    win_rate_count += 1
            
            if win_rate_count > 0:
                win_rates.append(win_rate_sum / win_rate_count)
        
        summary = {
            'total_checkpoints': len(checkpoints),
            'first_timestep': checkpoints[0]['timestep'],
            'last_timestep': checkpoints[-1]['timestep'],
            'best_elo_rating': max(elo_ratings) if elo_ratings else 0,
            'avg_elo_rating': np.mean(elo_ratings) if elo_ratings else 0,
            'best_checkpoint_path': self.best_checkpoints[0][1] if self.best_checkpoints else None,
            'latest_checkpoint_path': self.latest_checkpoints[0][1] if self.latest_checkpoints else None,
        }
        
        if win_rates:
            summary['best_win_rate'] = max(win_rates)
            summary['avg_win_rate'] = np.mean(win_rates)
        
        return summary 