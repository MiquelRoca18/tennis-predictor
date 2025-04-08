"""
Tennis Data Synchronization

This module provides functionality for automatic model updates when new data 
arrives, version management for models, and tools to monitor model performance
over time.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
import os
import json
import time
import shutil
from datetime import datetime, timedelta
import sqlite3
import joblib
import matplotlib.pyplot as plt
import threading
import queue
import hashlib
import schedule

# Configure logger
logger = logging.getLogger(__name__)

class TennisDataSynchronizer:
    """
    Class for managing automatic model updates and version control.
    
    This class provides functionality for:
    - Detecting new data and triggering model updates
    - Managing model versions and history
    - Tracking model performance over time
    - Automatic model refreshing based on schedules or performance thresholds
    
    Attributes:
        data_dir (str): Directory containing tennis data
        models_dir (str): Directory for storing model versions
        history_db (str): Path to SQLite database for tracking history
        current_model_path (str): Path to the current active model
        update_interval (int): Time in seconds between update checks
        model_trainer (object): Object that provides model training functionality
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        models_dir: str = 'models',
        history_db: str = 'model_history.db',
        current_model_path: str = 'models/current_model.joblib',
        update_interval: int = 86400,  # 24 hours
        model_trainer: Optional[Any] = None
    ):
        """
        Initialize the tennis data synchronizer.
        
        Args:
            data_dir: Directory containing tennis data
            models_dir: Directory for storing model versions
            history_db: Path to SQLite database for tracking history
            current_model_path: Path to the current active model
            update_interval: Time in seconds between update checks
            model_trainer: Object that provides model training functionality
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.history_db = history_db
        self.current_model_path = current_model_path
        self.update_interval = update_interval
        self.model_trainer = model_trainer
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Set up data tracking
        self.data_checksums = {}
        self.last_update_check = datetime.now()
        
        # For background processing
        self.update_thread = None
        self.stop_event = threading.Event()
        self.update_queue = queue.Queue()
        
        logger.info("Initialized TennisDataSynchronizer")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for tracking model history."""
        try:
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            # Create model versions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT UNIQUE,
                model_path TEXT,
                created_at TIMESTAMP,
                training_data_hash TEXT,
                config_hash TEXT,
                is_active INTEGER,
                description TEXT
            )
            ''')
            
            # Create model metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                evaluation_date TIMESTAMP,
                dataset_name TEXT,
                FOREIGN KEY (version_id) REFERENCES model_versions(version_id)
            )
            ''')
            
            # Create data updates table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_file TEXT,
                update_time TIMESTAMP,
                file_hash TEXT,
                file_size INTEGER,
                records_added INTEGER,
                processed INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized database at {self.history_db}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def scan_data_directory(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the data directory for new or modified files.
        
        Returns:
            Dictionary of file info keyed by file path
        """
        file_info = {}
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.csv', '.json')):
                    file_path = os.path.join(root, file)
                    
                    # Get file stats
                    stats = os.stat(file_path)
                    modified_time = datetime.fromtimestamp(stats.st_mtime)
                    size = stats.st_size
                    
                    # Calculate hash (can be expensive for large files)
                    file_hash = self.calculate_file_hash(file_path)
                    
                    file_info[file_path] = {
                        'modified_time': modified_time,
                        'size': size,
                        'hash': file_hash
                    }
        
        return file_info
    
    def detect_data_changes(self) -> List[str]:
        """
        Detect changes in the data files since the last check.
        
        Returns:
            List of paths to files that have changed
        """
        # Scan current state
        current_files = self.scan_data_directory()
        
        # Find changes
        changed_files = []
        
        for file_path, info in current_files.items():
            # Check if file is new
            if file_path not in self.data_checksums:
                changed_files.append(file_path)
                logger.info(f"New file detected: {file_path}")
            
            # Check if file has changed
            elif info['hash'] != self.data_checksums.get(file_path, {}).get('hash'):
                changed_files.append(file_path)
                logger.info(f"Modified file detected: {file_path}")
        
        # Update checksums
        self.data_checksums = current_files
        self.last_update_check = datetime.now()
        
        return changed_files
    
    def log_data_update(self, file_path: str, records_added: int = 0) -> None:
        """
        Log a data update to the database.
        
        Args:
            file_path: Path to the updated file
            records_added: Number of records added
        """
        try:
            # Get file info
            file_info = self.data_checksums.get(file_path, {})
            file_hash = file_info.get('hash', '')
            file_size = file_info.get('size', 0)
            
            # Log to database
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO data_updates 
                (data_file, update_time, file_hash, file_size, records_added, processed)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (file_path, datetime.now(), file_hash, file_size, records_added, 0)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged update for {file_path}: {records_added} records added")
        except Exception as e:
            logger.error(f"Error logging data update: {e}")
    
    def register_model_version(
        self,
        model_path: str,
        training_data_hash: str,
        config_hash: str,
        description: str = "",
        set_active: bool = True
    ) -> str:
        """
        Register a new model version in the database.
        
        Args:
            model_path: Path to the model file
            training_data_hash: Hash of the training data
            config_hash: Hash of the configuration
            description: Description of the model version
            set_active: Whether to set this model as the active version
            
        Returns:
            Version ID of the new model
        """
        try:
            # Generate version ID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            version_id = f"model_{timestamp}"
            
            # Set previous active model to inactive if setting new active
            if set_active:
                conn = sqlite3.connect(self.history_db)
                cursor = conn.cursor()
                
                cursor.execute(
                    "UPDATE model_versions SET is_active = 0 WHERE is_active = 1"
                )
                
                conn.commit()
                conn.close()
            
            # Register new model
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT INTO model_versions 
                (version_id, model_path, created_at, training_data_hash, config_hash, is_active, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (version_id, model_path, datetime.now(), training_data_hash, config_hash, 1 if set_active else 0, description)
            )
            
            conn.commit()
            conn.close()
            
            # If setting as active, update the current model symlink
            if set_active:
                self.set_active_model(version_id)
            
            logger.info(f"Registered model version {version_id}")
            return version_id
        except Exception as e:
            logger.error(f"Error registering model version: {e}")
            return ""
    
    def set_active_model(self, version_id: str) -> bool:
        """
        Set a model version as the active one.
        
        Args:
            version_id: Version ID of the model to set as active
            
        Returns:
            Whether the operation succeeded
        """
        try:
            # Update database
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            # Set all models to inactive
            cursor.execute("UPDATE model_versions SET is_active = 0")
            
            # Set target model to active
            cursor.execute(
                "UPDATE model_versions SET is_active = 1 WHERE version_id = ?",
                (version_id,)
            )
            
            # Get model path
            cursor.execute(
                "SELECT model_path FROM model_versions WHERE version_id = ?",
                (version_id,)
            )
            result = cursor.fetchone()
            
            conn.commit()
            conn.close()
            
            if not result:
                logger.error(f"Model version {version_id} not found")
                return False
            
            model_path = result[0]
            
            # Update current model symlink or copy
            current_dir = os.path.dirname(self.current_model_path)
            os.makedirs(current_dir, exist_ok=True)
            
            # Remove existing symlink or file
            if os.path.exists(self.current_model_path):
                if os.path.islink(self.current_model_path):
                    os.unlink(self.current_model_path)
                else:
                    os.remove(self.current_model_path)
            
            # Try to create symlink, fall back to copy if not supported
            try:
                os.symlink(model_path, self.current_model_path)
            except (OSError, AttributeError):
                shutil.copy2(model_path, self.current_model_path)
            
            logger.info(f"Set model version {version_id} as active")
            return True
        except Exception as e:
            logger.error(f"Error setting active model: {e}")
            return False
    
    def log_model_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float],
        dataset_name: str = "validation"
    ) -> bool:
        """
        Log evaluation metrics for a model version.
        
        Args:
            version_id: Version ID of the model
            metrics: Dictionary of metric names and values
            dataset_name: Name of the dataset used for evaluation
            
        Returns:
            Whether the operation succeeded
        """
        try:
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            # Insert each metric
            for metric_name, metric_value in metrics.items():
                cursor.execute(
                    '''
                    INSERT INTO model_metrics 
                    (version_id, metric_name, metric_value, evaluation_date, dataset_name)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (version_id, metric_name, metric_value, datetime.now(), dataset_name)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged {len(metrics)} metrics for model {version_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging model metrics: {e}")
            return False
    
    def get_model_versions(self) -> pd.DataFrame:
        """
        Get all model versions from the database.
        
        Returns:
            DataFrame of model versions
        """
        try:
            conn = sqlite3.connect(self.history_db)
            
            query = """
            SELECT 
                v.version_id, 
                v.created_at, 
                v.is_active,
                v.description,
                COUNT(DISTINCT m.metric_name) as metric_count
            FROM 
                model_versions v
            LEFT JOIN 
                model_metrics m ON v.version_id = m.version_id
            GROUP BY 
                v.version_id
            ORDER BY 
                v.created_at DESC
            """
            
            versions_df = pd.read_sql_query(query, conn)
            conn.close()
            
            return versions_df
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return pd.DataFrame()
    
    def get_model_metrics(self, version_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get metrics for a specific model version or all versions.
        
        Args:
            version_id: Version ID of the model (or None for all versions)
            
        Returns:
            DataFrame of model metrics
        """
        try:
            conn = sqlite3.connect(self.history_db)
            
            if version_id:
                query = """
                SELECT 
                    version_id, 
                    metric_name, 
                    metric_value, 
                    evaluation_date,
                    dataset_name
                FROM 
                    model_metrics
                WHERE 
                    version_id = ?
                ORDER BY 
                    evaluation_date
                """
                metrics_df = pd.read_sql_query(query, conn, params=(version_id,))
            else:
                query = """
                SELECT 
                    version_id, 
                    metric_name, 
                    metric_value, 
                    evaluation_date,
                    dataset_name
                FROM 
                    model_metrics
                ORDER BY 
                    evaluation_date
                """
                metrics_df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            return metrics_df
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return pd.DataFrame()
    
    def compare_model_versions(
        self,
        version_ids: List[str],
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple model versions.
        
        Args:
            version_ids: List of version IDs to compare
            metric_names: List of metric names to compare (or None for all)
            
        Returns:
            DataFrame with comparison results
        """
        try:
            # Get metrics for all specified versions
            metrics_list = []
            
            for version_id in version_ids:
                version_metrics = self.get_model_metrics(version_id)
                metrics_list.append(version_metrics)
            
            if not metrics_list or all(df.empty for df in metrics_list):
                logger.warning("No metrics found for comparison")
                return pd.DataFrame()
            
            # Combine metrics
            combined_metrics = pd.concat(metrics_list)
            
            # Filter by metric names if specified
            if metric_names:
                combined_metrics = combined_metrics[combined_metrics['metric_name'].isin(metric_names)]
            
            # Pivot for comparison
            comparison = combined_metrics.pivot_table(
                index=['metric_name', 'dataset_name'],
                columns='version_id',
                values='metric_value',
                aggfunc='mean'
            )
            
            # Calculate differences
            if len(version_ids) > 1:
                for i in range(len(version_ids)):
                    for j in range(i+1, len(version_ids)):
                        v1 = version_ids[i]
                        v2 = version_ids[j]
                        
                        if v1 in comparison.columns and v2 in comparison.columns:
                            comparison[f'{v1} vs {v2}'] = comparison[v1] - comparison[v2]
                            comparison[f'{v1} vs {v2} (%)'] = (comparison[v1] / comparison[v2] - 1) * 100
            
            return comparison.reset_index()
        except Exception as e:
            logger.error(f"Error comparing model versions: {e}")
            return pd.DataFrame()
    
    def plot_metric_history(
        self,
        metric_name: str,
        dataset_name: Optional[str] = None,
        last_n_versions: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot the history of a specific metric across model versions.
        
        Args:
            metric_name: Name of the metric to plot
            dataset_name: Name of the dataset to filter by (or None for all)
            last_n_versions: Number of most recent versions to include (or None for all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            # Get all metrics
            metrics_df = self.get_model_metrics()
            
            # Filter by metric name
            metrics_df = metrics_df[metrics_df['metric_name'] == metric_name]
            
            # Filter by dataset name if specified
            if dataset_name:
                metrics_df = metrics_df[metrics_df['dataset_name'] == dataset_name]
            
            if metrics_df.empty:
                logger.warning(f"No metrics found for {metric_name}")
                return None
            
            # Get version creation times
            versions_df = self.get_model_versions()
            versions_df['created_at'] = pd.to_datetime(versions_df['created_at'])
            
            # Merge creation times
            metrics_df = pd.merge(
                metrics_df,
                versions_df[['version_id', 'created_at']],
                on='version_id',
                how='left'
            )
            
            # Sort by creation time
            metrics_df = metrics_df.sort_values('created_at')
            
            # Limit to last N versions if specified
            if last_n_versions:
                version_ids = metrics_df['version_id'].unique()
                if len(version_ids) > last_n_versions:
                    versions_to_keep = version_ids[-last_n_versions:]
                    metrics_df = metrics_df[metrics_df['version_id'].isin(versions_to_keep)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot by dataset name
            for dataset, group in metrics_df.groupby('dataset_name'):
                ax.plot(
                    group['created_at'], 
                    group['metric_value'], 
                    'o-', 
                    label=f'{dataset} dataset'
                )
            
            # Mark active version
            active_version = versions_df[versions_df['is_active'] == 1]['version_id'].values
            if len(active_version) > 0:
                active_metrics = metrics_df[metrics_df['version_id'] == active_version[0]]
                if not active_metrics.empty:
                    ax.scatter(
                        active_metrics['created_at'],
                        active_metrics['metric_value'],
                        s=100,
                        c='red',
                        marker='*',
                        label='Active model'
                    )
            
            # Add labels and title
            ax.set_xlabel('Model Version (Creation Date)')
            ax.set_ylabel(f'{metric_name.capitalize()} Value')
            ax.set_title(f'History of {metric_name.capitalize()} Across Model Versions')
            
            # Format x-axis
            fig.autofmt_xdate()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting metric history: {e}")
            return None
    
    def check_for_updates(self) -> bool:
        """
        Check for data updates and trigger model retraining if needed.
        
        Returns:
            Whether any updates were detected
        """
        # Detect changes in data files
        changed_files = self.detect_data_changes()
        
        if not changed_files:
            logger.info("No data updates detected")
            return False
        
        # Log updates
        for file_path in changed_files:
            self.log_data_update(file_path)
        
        # Queue an update job
        self.queue_model_update(changed_files)
        
        return True
    
    def queue_model_update(self, changed_files: List[str]) -> None:
        """
        Queue a model update job.
        
        Args:
            changed_files: List of files that have changed
        """
        # Add to queue
        self.update_queue.put({
            'files': changed_files,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Queued model update for {len(changed_files)} changed files")
        
        # Start background thread if not running
        if self.update_thread is None or not self.update_thread.is_alive():
            self.start_update_thread()
    
    def start_update_thread(self) -> None:
        """Start the background update thread."""
        # Reset stop event
        self.stop_event.clear()
        
        # Create and start thread
        self.update_thread = threading.Thread(
            target=self._update_worker,
            daemon=True
        )
        self.update_thread.start()
        
        logger.info("Started update worker thread")
    
    def stop_update_thread(self) -> None:
        """Stop the background update thread."""
        if self.update_thread and self.update_thread.is_alive():
            self.stop_event.set()
            self.update_thread.join(timeout=10)
            logger.info("Stopped update worker thread")
    
    def _update_worker(self) -> None:
        """Worker function for processing model updates."""
        while not self.stop_event.is_set():
            try:
                # Get update job with 1-second timeout
                try:
                    update_job = self.update_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the update
                logger.info(f"Processing model update for {len(update_job['files'])} files")
                
                # Skip if no model trainer
                if self.model_trainer is None:
                    logger.warning("No model trainer provided, cannot update model")
                    continue
                
                # Train new model
                try:
                    # Calculate data hash
                    data_hash = hashlib.sha256()
                    for file_path in update_job['files']:
                        file_hash = self.data_checksums.get(file_path, {}).get('hash', '')
                        data_hash.update(file_hash.encode())
                    
                    # Calculate config hash (if available)
                    config_hash = ''
                    if hasattr(self.model_trainer, 'config'):
                        config_str = json.dumps(self.model_trainer.config, sort_keys=True)
                        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
                    
                    # Train model
                    training_result = self.model_trainer.run_training_pipeline()
                    
                    if not training_result:
                        logger.error("Model training failed")
                        continue
                    
                    model, metrics = training_result
                    
                    # Save model to version-specific path
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    model_path = os.path.join(self.models_dir, f"model_{timestamp}.joblib")
                    
                    joblib.dump(model, model_path)
                    
                    # Register model version
                    version_id = self.register_model_version(
                        model_path=model_path,
                        training_data_hash=data_hash.hexdigest(),
                        config_hash=config_hash,
                        description=f"Automatic update based on {len(update_job['files'])} changed files",
                        set_active=True
                    )
                    
                    # Log metrics
                    if metrics:
                        self.log_model_metrics(version_id, metrics)
                    
                    logger.info(f"Successfully updated model to version {version_id}")
                
                except Exception as e:
                    logger.error(f"Error updating model: {e}")
                
                # Mark job as done
                self.update_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in update worker: {e}")
        
        logger.info("Update worker thread stopped")
    
    def schedule_regular_updates(self, time_str: str = "00:00") -> None:
        """
        Schedule regular model updates.
        
        Args:
            time_str: Time of day to run updates (HH:MM format)
        """
        # Schedule the update
        schedule.every().day.at(time_str).do(self.check_for_updates)
        
        logger.info(f"Scheduled daily model updates at {time_str}")
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(
            target=self._scheduler_worker,
            daemon=True
        )
        scheduler_thread.start()
    
    def _scheduler_worker(self) -> None:
        """Worker function for running scheduled tasks."""
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start_monitoring(self, interval: Optional[int] = None) -> None:
        """
        Start continuous monitoring for data updates.
        
        Args:
            interval: Interval in seconds between checks (or None to use default)
        """
        if interval:
            self.update_interval = interval
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True
        )
        monitoring_thread.start()
        
        logger.info(f"Started monitoring for data updates (interval: {self.update_interval}s)")
    
    def _monitoring_worker(self) -> None:
        """Worker function for continuous monitoring."""
        while not self.stop_event.is_set():
            # Check for updates
            self.check_for_updates()
            
            # Sleep until next check
            time.sleep(self.update_interval)
    
    def generate_version_report(self, version_id: Optional[str] = None) -> str:
        """
        Generate a report for a specific model version or the active version.
        
        Args:
            version_id: Version ID of the model (or None for active version)
            
        Returns:
            Markdown formatted report
        """
        try:
            # Get version info
            conn = sqlite3.connect(self.history_db)
            cursor = conn.cursor()
            
            if version_id:
                cursor.execute(
                    "SELECT * FROM model_versions WHERE version_id = ?",
                    (version_id,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM model_versions WHERE is_active = 1"
                )
            
            version_info = cursor.fetchone()
            conn.close()
            
            if not version_info:
                logger.warning(f"No version found for report")
                return "No model version found"
            
            # Extract version details
            version_id = version_info[1]
            created_at = version_info[3]
            description = version_info[7]
            
            # Get metrics
            metrics_df = self.get_model_metrics(version_id)
            
            # Generate report
            report = [
                f"# Model Version: {version_id}\n\n",
                f"Created: {created_at}\n\n",
                f"Description: {description}\n\n"
            ]
            
            # Add metrics section
            if not metrics_df.empty:
                report.append("## Performance Metrics\n\n")
                
                # Group by dataset
                for dataset, group in metrics_df.groupby('dataset_name'):
                    report.append(f"### {dataset.capitalize()} Dataset\n\n")
                    
                    # Create table
                    report.append("| Metric | Value |\n|--------|-------|\n")
                    
                    for _, row in group.iterrows():
                        metric_name = row['metric_name'].replace('_', ' ').title()
                        metric_value = row['metric_value']
                        
                        report.append(f"| {metric_name} | {metric_value:.4f} |\n")
                    
                    report.append("\n")
            
            # Add version history
            previous_versions = self.get_model_versions()
            if len(previous_versions) > 1:
                report.append("## Version History\n\n")
                
                # Create table
                report.append("| Version ID | Created | Metrics |\n|------------|---------|--------|\n")
                
                for _, row in previous_versions.iterrows():
                    v_id = row['version_id']
                    v_created = row['created_at']
                    v_metrics = row['metric_count']
                    v_active = "✓" if row['is_active'] == 1 else ""
                    
                    report.append(f"| {v_id} {v_active} | {v_created} | {v_metrics} |\n")
                
                report.append("\n")
            
            return "".join(report)
        except Exception as e:
            logger.error(f"Error generating version report: {e}")
            return f"Error generating report: {e}"