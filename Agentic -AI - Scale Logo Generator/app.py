#!/usr/bin/env python3
"""
Flask Backend for Agentic Logo Generation Web UI

This Flask application provides REST API endpoints to integrate the web UI
with the existing logo generation pipeline.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError
import threading
import time

# Import your existing pipeline components
try:
    from logo_pipeline import LogoGenerationPipeline, ProjectStorage
    from api_config import validate_config, get_api_key
    from examples import get_all_examples
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure all pipeline files are in the same directory as this Flask app")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for web UI

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Global variables to track generation jobs
generation_jobs = {}
job_status = {}

class GenerationJob:
    """Represents a logo generation job with status tracking."""
    
    def __init__(self, job_id: str, club_description: str, personal_vision: str = ""):
        self.job_id = job_id
        self.club_description = club_description
        self.personal_vision = personal_vision
        self.status = "initialized"
        self.progress = 0
        self.current_step = ""
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at = None
        self.project_dir = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "project_dir": self.project_dir
        }

def update_job_status(job_id: str, status: str, progress: int, step: str = ""):
    """Update the status of a generation job."""
    if job_id in generation_jobs:
        job = generation_jobs[job_id]
        job.status = status
        job.progress = progress
        job.current_step = step
        logger.info(f"Job {job_id}: {status} - {progress}% - {step}")

def run_generation_pipeline(job_id: str):
    """Run the logo generation pipeline in a separate thread."""
    try:
        job = generation_jobs[job_id]
        update_job_status(job_id, "running", 10, "Initializing pipeline...")
        
        # Create pipeline
        pipeline = LogoGenerationPipeline()
        
        update_job_status(job_id, "running", 25, "Generating logo designs...")
        
        # Run the pipeline with progress updates
        result = pipeline.run_pipeline(job.club_description, job.personal_vision)
        
        update_job_status(job_id, "running", 75, "Evaluating designs...")
        
        # Store results
        job.result = result
        job.project_dir = pipeline.storage.project_dir
        job.completed_at = datetime.now()
        
        update_job_status(job_id, "completed", 100, "Pipeline completed successfully!")
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed with error: {str(e)}")
        job = generation_jobs[job_id]
        job.error = str(e)
        job.completed_at = datetime.now()
        update_job_status(job_id, "failed", 0, f"Error: {str(e)}")

# API Routes

@app.route('/')
def index():
    """Serve the main web UI."""
    # Read the HTML file content
    html_file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Return a simple page with instructions
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agentic Logo Generator - Backend</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; }
                code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🎨 Agentic Logo Generator - Backend Running</h1>
            <div class="status">
                ✅ Flask backend is running successfully!<br>
                📡 API endpoints are available at: <code>http://localhost:5000/api/</code>
            </div>
            
            <h2>📋 Available API Endpoints</h2>
            <div class="endpoint"><strong>GET</strong> <code>/api/health</code> - Health check</div>
            <div class="endpoint"><strong>GET</strong> <code>/api/examples</code> - Get club examples</div>
            <div class="endpoint"><strong>POST</strong> <code>/api/generate</code> - Start logo generation</div>
            <div class="endpoint"><strong>GET</strong> <code>/api/status/&lt;job_id&gt;</code> - Check generation status</div>
            <div class="endpoint"><strong>GET</strong> <code>/api/results/&lt;job_id&gt;</code> - Get generation results</div>
            <div class="endpoint"><strong>POST</strong> <code>/api/config/test</code> - Test API configuration</div>
            
            <h2>🚀 Getting Started</h2>
            <ol>
                <li>Save the provided HTML UI as <code>index.html</code> in this directory</li>
                <li>Update the API endpoint in the UI to point to this backend</li>
                <li>Make sure your OpenAI API key is configured in <code>api_config.py</code></li>
                <li>Access the full UI at <code>http://localhost:5000/</code></li>
            </ol>
            
            <p><strong>Note:</strong> Make sure all your pipeline files (<code>logo_pipeline.py</code>, <code>api_config.py</code>, <code>examples.py</code>) are in the same directory as this Flask app.</p>
        </body>
        </html>
        """

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check if pipeline components are available
        pipeline_available = True
        try:
            from logo_pipeline import LogoGenerationPipeline
            from api_config import validate_config
        except ImportError:
            pipeline_available = False
        
        # Check API configuration
        config_valid = validate_config() if pipeline_available else False
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pipeline_available": pipeline_available,
            "config_valid": config_valid,
            "active_jobs": len([j for j in generation_jobs.values() if j.status == "running"])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/examples')
def get_examples():
    """Get all available club examples."""
    try:
        examples = get_all_examples()
        return jsonify({
            "success": True,
            "examples": examples
        })
    except Exception as e:
        logger.error(f"Error getting examples: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/config/test', methods=['POST'])
def test_config():
    """Test API configuration."""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({"success": False, "error": "API key is required"}), 400
        
        # Test OpenAI connection
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Test with a simple completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return jsonify({
                "success": True,
                "message": "API connection successful",
                "model_accessible": True
            })
            
        except Exception as api_error:
            return jsonify({
                "success": False,
                "error": f"API connection failed: {str(api_error)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error testing config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_logos():
    """Start logo generation process."""
    try:
        data = request.get_json()
        
        if not data:
            raise BadRequest("No data provided")
        
        club_description = data.get('club_description', '').strip()
        personal_vision = data.get('personal_vision', '').strip()
        
        if not club_description:
            raise BadRequest("Club description is required")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job
        job = GenerationJob(job_id, club_description, personal_vision)
        generation_jobs[job_id] = job
        
        # Start generation in background thread
        thread = threading.Thread(target=run_generation_pipeline, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started generation job {job_id}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Logo generation started",
            "status_url": f"/api/status/{job_id}"
        })
        
    except BadRequest as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error starting generation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """Get the status of a generation job."""
    try:
        if job_id not in generation_jobs:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        job = generation_jobs[job_id]
        return jsonify({
            "success": True,
            "job": job.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/results/<job_id>')
def get_job_results(job_id):
    """Get the results of a completed generation job."""
    try:
        if job_id not in generation_jobs:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        job = generation_jobs[job_id]
        
        if job.status != "completed":
            return jsonify({
                "success": False, 
                "error": "Job not completed yet",
                "status": job.status
            }), 400
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "results": job.result,
            "project_dir": job.project_dir
        })
        
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download/<job_id>/<file_type>')
def download_results(job_id, file_type):
    """Download results in various formats."""
    try:
        if job_id not in generation_jobs:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        job = generation_jobs[job_id]
        
        if job.status != "completed":
            return jsonify({"success": False, "error": "Job not completed"}), 400
        
        if file_type == "json":
            # Create JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logo_results_{timestamp}.json"
            
            # Create temporary file
            temp_path = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)
            
            with open(temp_path, 'w') as f:
                json.dump(job.result, f, indent=2)
            
            return send_file(temp_path, as_attachment=True, download_name=filename)
        
        else:
            return jsonify({"success": False, "error": "Unsupported file type"}), 400
            
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/logos/<job_id>/<logo_id>')
def get_logo_image(job_id, logo_id):
    """Serve logo images."""
    try:
        if job_id not in generation_jobs:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        job = generation_jobs[job_id]
        
        if not job.project_dir:
            return jsonify({"success": False, "error": "Project directory not found"}), 404
        
        # Look for logo image in project directory
        logo_dir = os.path.join(job.project_dir, "generated_logos", logo_id)
        
        if os.path.exists(logo_dir):
            # Find image file
            for file in os.listdir(logo_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    return send_from_directory(logo_dir, file)
        
        return jsonify({"success": False, "error": "Logo image not found"}), 404
        
    except Exception as e:
        logger.error(f"Error serving logo image: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/jobs')
def list_jobs():
    """List all generation jobs."""
    try:
        jobs_list = []
        for job_id, job in generation_jobs.items():
            jobs_list.append({
                "job_id": job_id,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            })
        
        return jsonify({
            "success": True,
            "jobs": jobs_list,
            "total": len(jobs_list)
        })
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a generation job."""
    try:
        if job_id not in generation_jobs:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        job = generation_jobs[job_id]
        
        # Only allow deletion of completed or failed jobs
        if job.status == "running":
            return jsonify({"success": False, "error": "Cannot delete running job"}), 400
        
        # Remove job from memory
        del generation_jobs[job_id]
        
        # Optionally clean up project directory
        # (commented out for safety - you might want to keep the files)
        # if job.project_dir and os.path.exists(job.project_dir):
        #     shutil.rmtree(job.project_dir)
        
        return jsonify({"success": True, "message": "Job deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting job: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

@app.errorhandler(BadRequest)
def bad_request(error):
    return jsonify({"success": False, "error": str(error)}), 400

# Utility functions
def cleanup_old_jobs():
    """Clean up old completed jobs (run periodically)."""
    current_time = datetime.now()
    jobs_to_remove = []
    
    for job_id, job in generation_jobs.items():
        if job.completed_at:
            time_diff = current_time - job.completed_at
            # Remove jobs older than 24 hours
            if time_diff.total_seconds() > 24 * 3600:
                jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        logger.info(f"Cleaning up old job: {job_id}")
        del generation_jobs[job_id]

def setup_directories():
    """Create necessary directories."""
    directories = ['temp', 'uploads', 'static']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == '__main__':
    # Setup
    setup_directories()
    
    # Check configuration
    try:
        config_valid = validate_config()
        if config_valid:
            logger.info("✅ Configuration validated successfully")
        else:
            logger.warning("⚠️ Configuration validation failed - some features may not work")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
    
    # Start cleanup scheduler in background
    def periodic_cleanup():
        while True:
            time.sleep(3600)  # Run every hour
            cleanup_old_jobs()
    
    cleanup_thread = threading.Thread(target=periodic_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Start Flask app
    logger.info("🚀 Starting Agentic Logo Generator Backend...")
    logger.info("📡 Backend will be available at: http://localhost:5000")
    logger.info("🎨 Web UI will be available at: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )