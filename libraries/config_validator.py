"""Configuration validation for recruitment advertisement search."""
from typing import Dict, Any
import json
from logging_config import setup_logging

logger = setup_logging("config_validator")

class ConfigValidator:
    """Validates search configuration structure and values."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate the search configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check if run_configs exists and is a list
            if 'run_configs' not in config:
                logger.error("Missing 'run_configs' in configuration")
                return False
            
            if not isinstance(config['run_configs'], list):
                logger.error("'run_configs' must be a list")
                return False
            
            # Validate each run configuration
            for run_config in config['run_configs']:
                if not isinstance(run_config, dict):
                    logger.error("Each run configuration must be a dictionary")
                    return False
                
                # Check required fields
                if 'id' not in run_config:
                    logger.error("Missing 'id' in run configuration")
                    return False
                
                if 'days_back' not in run_config:
                    logger.error("Missing 'days_back' in run configuration")
                    return False
                
                # Validate days_back is a positive integer
                if not isinstance(run_config['days_back'], int) or run_config['days_back'] <= 0:
                    logger.error("'days_back' must be a positive integer")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return False 