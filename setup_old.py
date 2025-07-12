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
    
    def setup_virtual_environment(self, force: bool = False) -> bool:
        """Set up Python virtual environment"""
        logger.info("🐍 Setting up virtual environment...")
        
        if self.venv_dir.exists():
            if force:
                logger.info("Removing existing virtual environment...")
                subprocess.run([sys.executable, "-m", "shutil", "rmtree", str(self.venv_dir)])
            else:
                logger.info("✅ Virtual environment already exists")
                return True
        
        try:
            # Create virtual environment
            subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_dir)])
            logger.info("✅ Virtual environment created successfully")
            
            # Get activation command
            if os.name == 'nt':  # Windows
                activate_cmd = str(self.venv_dir / "Scripts" / "activate")
                python_exe = str(self.venv_dir / "Scripts" / "python")
            else:  # Unix/Linux/Mac
                activate_cmd = f"source {self.venv_dir}/bin/activate"
                python_exe = str(self.venv_dir / "bin" / "python")
            
            logger.info(f"� To activate: {activate_cmd}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """Install required packages"""
        logger.info("📦 Installing dependencies...")
        
        # Get Python executable from venv if it exists
        if self.venv_dir.exists():
            if os.name == 'nt':
                python_exe = str(self.venv_dir / "Scripts" / "python")
                pip_exe = str(self.venv_dir / "Scripts" / "pip")
            else:
                python_exe = str(self.venv_dir / "bin" / "python")
                pip_exe = str(self.venv_dir / "bin" / "pip")
        else:
            python_exe = sys.executable
            pip_exe = "pip"
        
        try:
            # Upgrade pip first
            subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install requirements
            cmd = [pip_exe, "install", "-r", "requirements.txt"]
            if upgrade:
                cmd.append("--upgrade")
            
            subprocess.check_call(cmd)
            logger.info("✅ Dependencies installed successfully")
            
            # Install additional development dependencies
            dev_packages = [
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0", 
                "pytest-mock>=3.10.0",
                "black>=23.0.0",
                "flake8>=6.0.0",
                "mypy>=1.0.0"
            ]
            
            for package in dev_packages:
                try:
                    subprocess.check_call([pip_exe, "install", package], 
                                        stdout=subprocess.DEVNULL, 
                                        stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    logger.warning(f"Failed to install optional package: {package}")
            
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
    
    def generate_test_data(self) -> bool:
        """Generate test data"""
        logger.info("🧪 Generating test data...")
        
        try:
            # Create test data directory
            test_data_dir = self.test_dir / "test_data"
            test_data_dir.mkdir(exist_ok=True)
            
            # Generate test data
            from tests.test_factory import TestDataFactory
            factory = TestDataFactory()
            factory.save_test_data(str(test_data_dir))
            
            logger.info("✅ Test data generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate test data: {e}")
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
    
    def run_basic_tests(self) -> bool:
        """Run basic system tests"""
        logger.info("🧪 Running basic tests...")
        
        try:
            # Run a subset of unit tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_data_preprocessor.py::TestDataPreprocessor::test_create_test_data",
                "-v"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Basic tests passed")
                return True
            else:
                logger.error("❌ Basic tests failed")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return False
    
    def setup_demo_environment(self) -> bool:
        """Set up demo environment"""
        logger.info("🎭 Setting up demo environment...")
        
        try:
            # Create demo script
            demo_script = self.root_dir / "demo.py"
            
            demo_content = '''#!/usr/bin/env python3
"""
Demo script for AI Recommendation System
"""

import streamlit as st
from data.preprocessor import DataPreprocessor
from models.hybrid_recommender import HybridRecommender

def main():
    st.title("🍽️ AI Restaurant Recommendation Demo")
    
    # Load data
    with st.spinner("Loading data..."):
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        preprocessor.preprocess_restaurants()
        data = preprocessor.get_processed_data()
    
    # Train model
    with st.spinner("Training models..."):
        recommender = HybridRecommender()
        recommender.fit(data)
    
    # Demo interface
    st.success("System ready! Run 'streamlit run app.py' for full interface")

if __name__ == "__main__":
    main()
'''
            
            with open(demo_script, 'w') as f:
                f.write(demo_content)
            
            logger.info("✅ Demo environment ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup demo: {e}")
            return False
    
    def run_full_setup(self, force_env: bool = False, skip_tests: bool = False, 
                      interactive: bool = True) -> bool:
        """Run complete system setup"""
        logger.info("🚀 Starting full system setup...")
        
        success_count = 0
        total_steps = len(self.setup_steps)
        
        # Check Python version first
        if not self.check_python_version():
            return False
        
        # Run setup steps
        setup_functions = [
            (lambda: self.setup_virtual_environment(force_env), 'environment'),
            (lambda: self.install_dependencies(), 'dependencies'),
            (lambda: self.setup_environment_config(interactive), 'env_config'),
            (lambda: self.generate_sample_data(), 'data'),
            (lambda: self.generate_test_data(), 'test_data'),
            (lambda: self.validate_installation(), 'validation'),
            (lambda: self.run_basic_tests() if not skip_tests else True, 'tests'),
            (lambda: self.setup_demo_environment(), 'demo')
        ]
        
        for setup_func, step_name in setup_functions:
            step_desc = self.setup_steps[step_name]
            logger.info(f"\\n📋 Step: {step_desc}")
            
            if setup_func():
                success_count += 1
                logger.info(f"✅ Completed: {step_desc}")
            else:
                logger.error(f"❌ Failed: {step_desc}")
                
                if step_name in ['environment', 'dependencies']:
                    logger.error("Critical step failed. Aborting setup.")
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
        llm_provider = "github"
        if not github_token:
            print("⚠️  Warning: No GitHub token provided. LLM features will be disabled.")
    else:
        print("⚠️  LLM features will be disabled.")
    
    env_content = f"""# GitHub Token for accessing marketplace models API
GITHUB_TOKEN={github_token}

# Hugging Face Token for accessing HF models
HUGGINGFACE_TOKEN={hf_token}

# LLM Configuration
LLM_MODEL=microsoft/DialoGPT-medium
LLM_PROVIDER={llm_provider}  # Options: github, huggingface, openai
MAX_TOKENS=4000
TEMPERATURE=0.7

# Recommendation System Settings
USE_LLM_ENHANCEMENT={str(bool(github_token or hf_token)).lower()}
LLM_EXPLANATION_ENABLED={str(bool(github_token or hf_token)).lower()}
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def verify_installation():
    """Verify that the installation is working"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import pandas
        import numpy
        import streamlit
        import plotly
        import smolagents
        print("✅ Core packages imported successfully")
        
        # Check if .env file exists and has required variables
        from dotenv import load_dotenv
        load_dotenv()
        
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            print("✅ GitHub token configured")
        else:
            print("⚠️  GitHub token not configured - LLM features will be disabled")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def run_system():
    """Run the recommendation system"""
    print("🚀 Starting the AI Recommendation System...")
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the system: {e}")
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Setup and run AI Recommendation System")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--run", action="store_true", help="Run the system")
    parser.add_argument("--all", action="store_true", help="Do all setup steps and run")
    
    args = parser.parse_args()
    
    print("🍽️  AI Recommendation System Setup")
    print("=" * 40)
    
    if args.all or args.install:
        if not install_requirements():
            sys.exit(1)
    
    if args.all or args.setup:
        if not setup_environment():
            sys.exit(1)
    
    if args.all or args.verify:
        if not verify_installation():
            sys.exit(1)
    
    if args.all or args.run:
        run_system()
    
    if not any(vars(args).values()):
        print("Use --help to see available options")
        print("\nQuick start: python setup.py --all")

if __name__ == "__main__":
    main()
