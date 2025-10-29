"""
Configuration settings for EcoDrive Simulator backend
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ecodrive-simulator-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # Server settings
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # Model paths
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    
    # FastF1 cache settings
    FASTF1_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'fastf1_cache')
    
    # Simulation settings
    MAX_SIMULATION_TIME = float(os.environ.get('MAX_SIMULATION_TIME', 300.0))
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))  # Use PORT from environment (common in cloud platforms)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

