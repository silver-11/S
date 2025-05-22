import os
from flask import Flask, jsonify, make_response, request, redirect, url_for, send_from_directory, g
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from our new modules
import config
from db_manager import DatabaseManager
from models import load_clip_classification_model, load_gemini_model, load_timesformer_model
from auth_routes import auth_bp
from video_routes import video_bp

# Initialize Flask App
app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
CORS(app, 
     resources={r"/api/*": {"origins": ["https://scene-solver.vercel.app", "http://localhost:3000"]}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

# Apply configurations from config.py
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.secret_key = config.SECRET_KEY

# Initialize Database Manager
# This instance will be used by the auth_required decorator and potentially by blueprints if passed correctly.
db = DatabaseManager()

# Create necessary directories
config.create_directories()

# Load Models
print(f"Using device: {config.DEVICE}")
if config.DEVICE.type == 'cuda':
    import torch # Import torch here if not already imported, for get_device_name
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

load_clip_classification_model() # Loads into global variables in models.py
load_gemini_model() # Loads into global variables in models.py
load_timesformer_model() # Loads into global variables in models.py

# --- Authentication Decorator ---
def auth_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_token = request.cookies.get('session_token')
        if not session_token:
            return jsonify({"success": False, "message": "Authentication required. No session token."}), 401
        
        user_id = db.validate_session(session_token)
        if not user_id:
            response = make_response(jsonify({"success": False, "message": "Invalid or expired session."}), 401)
            # Clear the invalid cookie
            response.delete_cookie('session_token', httponly=True, samesite='Lax')
            return response
        
        user_info = db.get_user_info(user_id)
        if not user_info:
             response = make_response(jsonify({"success": False, "message": "Could not retrieve user details."}), 401)
             response.delete_cookie('session_token', httponly=True, samesite='Lax')
             return response
        
        # Make user info available for the duration of the request
        g.user = user_info # Using Flask's g for request-bound context
        request.user = user_info # Also attaching to request for easier access in route if not using g
        
        return func(*args, **kwargs)
    return wrapper

# Apply the decorator to specific routes within blueprints if needed, or to the whole blueprint.
# For simplicity, routes in blueprints that need protection will be wrapped individually or rely on a before_request hook.
# The /api/me and /api/upload, /api/caption routes are designed to be protected.

# Register Blueprints
app.register_blueprint(auth_bp) # Auth routes like /api/login, /api/register, etc.
app.register_blueprint(video_bp) # Video routes like /api/upload, /api/caption

# Move the before_request decorator here, after blueprint registration
@app.before_request
def before_request():
    # Skip HTTPS redirect in development mode
    if request.endpoint and 'static' not in request.endpoint:
        if not request.is_secure and app.config.get('ENV') == 'production':
            url = request.url.replace('http://', 'https://', 1)
            return redirect(url, code=301)

# Apply auth_required to protected routes in blueprints
# This is a way to apply it if not done within the blueprint. 
# For this setup, it's assumed that relevant routes in auth_routes.py and video_routes.py
# will be decorated or the blueprint itself can have a before_request hook.
# Example of explicit decoration if not done in blueprint:
# app.view_functions['auth_bp.current_user_info'] = auth_required(app.view_functions['auth_bp.current_user_info'])
# app.view_functions['video_bp.upload_video'] = auth_required(app.view_functions['video_bp.upload_video'])
# app.view_functions['video_bp.caption_video'] = auth_required(app.view_functions['video_bp.caption_video'])
# A more common way is to apply decorator directly in blueprint routes, or use `Blueprint.before_request`.
# The current auth_routes.py and video_routes.py have placeholders for where `auth_required` would be applied.
# Let's assume the `auth_required` decorator will be added to the individual routes within the blueprint files for clarity.
# Re-checking `auth_routes.py` and `video_routes.py` to ensure they are expecting this setup.
# For the `/api/me` route in `auth_routes.py`, it should use the `g.user` or `request.user`.
# For `/api/upload` and `/api/caption` in `video_routes.py`, they are protected as well.

# Decorate specific blueprint routes after registration:
# This ensures the app context and `db` instance are available to the decorator.
# Note: The view function name is `blueprint_name.function_name`
if 'auth_bp.current_user_info' in app.view_functions:
    app.view_functions['auth_bp.current_user_info'] = auth_required(app.view_functions['auth_bp.current_user_info'])
if 'video_bp.upload_video' in app.view_functions:
    app.view_functions['video_bp.upload_video'] = auth_required(app.view_functions['video_bp.upload_video'])
if 'video_bp.caption_video' in app.view_functions:
    app.view_functions['video_bp.caption_video'] = auth_required(app.view_functions['video_bp.caption_video'])


# --- Static File Serving and Index Route ---
@app.route('/')
def index():
    # This route can serve as a basic landing page or redirect logic if needed.
    # However, the catch-all `serve_react` handles serving index.html for the React app.
    # If direct access to '/' should also serve React app, this is okay.
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Main execution
if __name__ == '__main__':
    # Enable debug mode and auto-reloader with more verbose output
    app.run(
        debug=True,
        use_reloader=True,
        host='127.0.0.1',
        port=5000,
        reloader_type='stat'  # Use stat reloader for better performance
    ) 