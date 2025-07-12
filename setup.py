#!/usr/bin/env python3
"""
Advanced Setup Script for AI Recommendation System

Provides comprehensive system setup including dependency installation,
environment configuration, data generation, testing, and validation.
"""

import os
import sys
import subprocess
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SystemSetup:
    """Comprehensive system setup and configuration"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.venv_dir = self.root_dir / "venv"
        self.data_dir = self.root_dir / "data"
        self.test_dir = self.root_dir / "tests"
        
        self.setup_steps = {
            'environment': 'Set up Python virtual environment',
            'dependencies': 'Install Python dependencies',
            'env_config': 'Configure environment variables',
            'data': 'Generate sample data',
            'test_data': 'Generate test data',
            'validation': 'Validate system installation',
            'tests': 'Run basic tests',
            'demo': 'Set up demo environment'
        }
        
    def print_banner(self):
        """Print setup banner"""
        print("=" * 70)
        print("🤖 AI-Powered Restaurant Recommendation System Setup")
        print("   Enhanced with LLM Intelligence & Emotional AI")
        print("=" * 70)
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            logger.error(f"Python {min_version[0]}.{min_version[1]}+ required, got {current_version[0]}.{current_version[1]}")
            return False
        
        logger.info(f"✅ Python {current_version[0]}.{current_version[1]} detected")
        return True
    
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """Install required packages"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
            if upgrade:
                cmd.append("--upgrade")
            
            subprocess.check_call(cmd)
            logger.info("✅ Dependencies installed successfully")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_environment_config(self, interactive: bool = True) -> bool:
        """Setup environment configuration"""
        logger.info("🔧 Setting up environment configuration...")
        
        env_file = self.root_dir / ".env"
        
        if env_file.exists():
            if interactive:
                overwrite = input("📄 .env file exists. Overwrite? (y/N): ").strip().lower()
                if overwrite != 'y':
                    logger.info("✅ Using existing .env file")
                    return True
        
        env_config = {}
        
        if interactive:
            logger.info("\\n🤖 LLM Configuration:")
            logger.info("Choose your LLM provider for enhanced recommendations:")
            logger.info("1. Hugging Face (Recommended - Free)")
            logger.info("2. GitHub Marketplace Models")
            logger.info("3. OpenAI (Requires API key)")
            logger.info("4. Skip LLM setup")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                hf_token = input("Enter Hugging Face token (optional): ").strip()
                env_config.update({
                    'LLM_PROVIDER': 'huggingface',
                    'HUGGINGFACE_TOKEN': hf_token,
                    'USE_LLM_ENHANCEMENT': 'true',
                    'LLM_MODEL': 'microsoft/DialoGPT-medium'
                })
            elif choice == "2":
                github_token = input("Enter GitHub token: ").strip()
                env_config.update({
                    'LLM_PROVIDER': 'github',
                    'GITHUB_TOKEN': github_token,
                    'USE_LLM_ENHANCEMENT': 'true'
                })
            elif choice == "3":
                openai_key = input("Enter OpenAI API key: ").strip()
                env_config.update({
                    'LLM_PROVIDER': 'openai',
                    'OPENAI_API_KEY': openai_key,
                    'USE_LLM_ENHANCEMENT': 'true'
                })
            else:
                env_config.update({
                    'USE_LLM_ENHANCEMENT': 'false'
                })
            
            # Emotional AI configuration
            logger.info("\\n💭 Emotional AI Configuration:")
            emotional_ai = input("Enable Emotional AI features? (Y/n): ").strip().lower()
            env_config['USE_EMOTIONAL_RECOMMENDATIONS'] = 'true' if emotional_ai != 'n' else 'false'
            
            # Optional API keys
            logger.info("\\n🌐 Optional External APIs:")
            weather_key = input("Weather API key (optional): ").strip()
            if weather_key:
                env_config['WEATHER_API_KEY'] = weather_key
        
        else:
            # Non-interactive setup with defaults
            env_config = {
                'LLM_PROVIDER': 'huggingface',
                'USE_LLM_ENHANCEMENT': 'true',
                'USE_EMOTIONAL_RECOMMENDATIONS': 'true',
                'LLM_MODEL': 'microsoft/DialoGPT-medium',
                'MAX_TOKENS': '4000',
                'TEMPERATURE': '0.7'
            }
        
        # Add default configuration
        default_config = {
            'MAX_TOKENS': '4000',
            'TEMPERATURE': '0.7',
            'LLM_EXPLANATION_ENABLED': 'true',
            'EMOTIONAL_CACHE_TTL': '3600',
            'EMOTIONAL_WEIGHT': '0.3'
        }
        
        for key, value in default_config.items():
            if key not in env_config:
                env_config[key] = value
        
        # Write .env file
        try:
            with open(env_file, 'w') as f:
                f.write("# AI Recommendation System Configuration\\n")
                f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
                
                for key, value in env_config.items():
                    f.write(f"{key}={value}\\n")
            
            logger.info("✅ Environment configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create .env file: {e}")
            return False
    
    def generate_sample_data(self) -> bool:
        """Generate sample data for the system"""
        logger.info("📊 Generating sample data...")
        
        try:
            # Create data directories
            self.data_dir.mkdir(exist_ok=True)
            (self.data_dir / "emotional").mkdir(parents=True, exist_ok=True)
            
            # Generate data using test factory
            from tests.test_factory import TestDataFactory
            
            factory = TestDataFactory()
            
            # Generate and save sample data
            users = factory.create_users(200)
            restaurants = factory.create_restaurants(500)
            ratings = factory.create_ratings(users, restaurants, 5000)
            reviews = factory.create_reviews(ratings)
            
            # Save to CSV files
            users.to_csv(self.data_dir / "users.csv", index=False)
            restaurants.to_csv(self.data_dir / "restaurants.csv", index=False)
            ratings.to_csv(self.data_dir / "ratings.csv", index=False)
            reviews.to_csv(self.data_dir / "reviews.csv", index=False)
            
            # Generate emotional mappings
            emotional_mappings = factory.create_emotional_mappings()
            with open(self.data_dir / "emotional" / "emotion_mappings.json", 'w') as f:
                json.dump(emotional_mappings, f, indent=2)
            
            logger.info("✅ Sample data generated successfully")
            logger.info(f"   - {len(users)} users")
            logger.info(f"   - {len(restaurants)} restaurants")
            logger.info(f"   - {len(ratings)} ratings")
            logger.info(f"   - {len(reviews)} reviews")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sample data: {e}")
            return False
    
    def validate_installation(self) -> bool:
        """Validate system installation"""
        logger.info("🔍 Validating installation...")
        
        validation_steps = [
            ("Python modules", self._validate_python_modules),
            ("Data files", self._validate_data_files),
            ("Configuration", self._validate_configuration),
            ("Models", self._validate_models)
        ]
        
        all_valid = True
        
        for step_name, validator in validation_steps:
            try:
                if validator():
                    logger.info(f"   ✅ {step_name}")
                else:
                    logger.error(f"   ❌ {step_name}")
                    all_valid = False
            except Exception as e:
                logger.error(f"   ❌ {step_name}: {e}")
                all_valid = False
        
        if all_valid:
            logger.info("✅ Installation validation successful")
        else:
            logger.error("❌ Installation validation failed")
        
        return all_valid
    
    def _validate_python_modules(self) -> bool:
        """Validate Python modules can be imported"""
        required_modules = [
            'pandas', 'numpy', 'sklearn', 'streamlit',
            'nltk', 'textblob', 'plotly'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                logger.error(f"Missing module: {module}")
                return False
        
        return True
    
    def _validate_data_files(self) -> bool:
        """Validate data files exist"""
        required_files = [
            'data/users.csv',
            'data/restaurants.csv',
            'data/ratings.csv',
            'data/reviews.csv'
        ]
        
        for file_path in required_files:
            if not (self.root_dir / file_path).exists():
                logger.error(f"Missing data file: {file_path}")
                return False
        
        return True
    
    def _validate_configuration(self) -> bool:
        """Validate configuration files"""
        env_file = self.root_dir / ".env"
        if not env_file.exists():
            logger.error("Missing .env file")
            return False
        
        return True
    
    def _validate_models(self) -> bool:
        """Validate model modules can be imported"""
        model_modules = [
            'models.hybrid_recommender',
            'models.collaborative_filtering',
            'models.content_based_filtering',
            'models.sentiment_analyzer'
        ]
        
        for module in model_modules:
            try:
                __import__(module)
            except ImportError as e:
                logger.error(f"Cannot import {module}: {e}")
                return False
        
        return True
    
    def run_full_setup(self, interactive: bool = True) -> bool:
        """Run complete system setup"""
        logger.info("🚀 Starting full system setup...")
        
        success_count = 0
        total_steps = 4  # dependencies, env_config, data, validation
        
        # Check Python version first
        if not self.check_python_version():
            return False
        
        # Run setup steps
        setup_functions = [
            (lambda: self.install_dependencies(), 'Install dependencies'),
            (lambda: self.setup_environment_config(interactive), 'Configure environment'),
            (lambda: self.generate_sample_data(), 'Generate sample data'),
            (lambda: self.validate_installation(), 'Validate installation')
        ]
        
        for setup_func, step_desc in setup_functions:
            logger.info(f"\\n📋 Step: {step_desc}")
            
            if setup_func():
                success_count += 1
                logger.info(f"✅ Completed: {step_desc}")
            else:
                logger.error(f"❌ Failed: {step_desc}")
                break
        
        # Final summary
        logger.info(f"\\n{'='*50}")
        logger.info(f"SETUP SUMMARY: {success_count}/{total_steps} steps completed")
        
        if success_count == total_steps:
            logger.info("🎉 Setup completed successfully!")
            logger.info("\\n📋 Next steps:")
            logger.info("   1. Run 'streamlit run app.py' to start the application")
            logger.info("   2. Run 'python run_tests.py' to execute full test suite")
            logger.info("   3. Check README.md for detailed usage instructions")
            return True
        else:
            logger.error("❌ Setup incomplete. Please check errors above.")
            return False

def main():
    """Main setup entry point"""
    parser = argparse.ArgumentParser(description='AI Recommendation System Setup')
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete setup')
    
    parser.add_argument('--dependencies', action='store_true',
                       help='Install dependencies only')
    
    parser.add_argument('--config', action='store_true',
                       help='Setup configuration only')
    
    parser.add_argument('--data', action='store_true',
                       help='Generate sample data only')
    
    parser.add_argument('--validate', action='store_true',
                       help='Validate installation only')
    
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode')
    
    args = parser.parse_args()
    
    setup = SystemSetup()
    setup.print_banner()
    
    # Default to full setup if no specific action
    if not any([args.dependencies, args.config, args.data, args.validate]):
        args.all = True
    
    interactive = not args.non_interactive
    success = True
    
    try:
        if args.all:
            success = setup.run_full_setup(interactive=interactive)
        else:
            if args.dependencies:
                success &= setup.install_dependencies()
            
            if args.config:
                success &= setup.setup_environment_config(interactive=interactive)
            
            if args.data:
                success &= setup.generate_sample_data()
            
            if args.validate:
                success &= setup.validate_installation()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\\n⏹ Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\\n💥 Setup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
