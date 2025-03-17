from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import openai
import os

from PyPDF2 import PdfReader
import pandas as pd
from docx import Document

app = Flask(__name__)
openai.api_key = 'API key comes here'

# Flask-Login setup
app.secret_key = 'your-secret-key'  # Used to sign session cookies (Change this for production)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
#class User(UserMixin):
    #def __init__(self, id):
        #self.id = id
        
# User model with role
class User(UserMixin):
    def __init__(self, id, role):
        self.id = id
        self.role = role  # Add the role attribute

# Simple in-memory user storage
#users = {'testuser': {'password': 'password123'}}  # Example: username -> password (this should be hashed in production)
# Updated user storage with roles
users = {
    'Sakthi': {'password': 'password123', 'role': 'admin'},
    'User': {'password': 'password123', 'role': 'basic'}
}


# Folder where the documents are stored
# DOCUMENT_FOLDER = '/Users/ssathiyaseelan/Desktop/Sakthi/python/POC_22_07/Document_Search_BOT/upload'
DOCUMENT_FOLDER = '/Users/ssathiyaseelan/Desktop/python/POC_22_07/Document_Search_BOT/upload'

os.makedirs(DOCUMENT_FOLDER, exist_ok=True)  # Create the uploads folder if it doesn't exist

# Folder where uploaded files are stored
UPLOAD_FOLDER = DOCUMENT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only specific file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'xls'}

# Function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

documents = {}

# Helper function to load documents
def load_documents():
    global documents
    documents = {}
    for filename in os.listdir(DOCUMENT_FOLDER):
        filepath = os.path.join(DOCUMENT_FOLDER, filename)
        if filename.endswith('.pdf'):
            with open(filepath, 'rb') as file:
                text = extract_text_from_pdf(file)
                documents[filename] = text
        elif filename.endswith('.docx'):
            with open(filepath, 'rb') as file:
                text = extract_text_from_docx(file)
                documents[filename] = text
                
         # Process Excel files (XLSX)
        elif filename.endswith('.xlsx'):
            with open(filepath, 'rb') as file:
                text = extract_text_from_excel(file)
                documents[filename] = text
        
        # Process Excel files (XLS)
        elif filename.endswith('.xls'):
            with open(filepath, 'rb') as file:
                text = extract_text_from_excel(file)
                documents[filename] = text
    print(f"Loaded {len(documents)} documents.")

# Helper function to extract text from PDF files
def extract_text_from_pdf(file):
    from PyPDF2 import PdfReader
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Helper function to extract text from DOCX files
def extract_text_from_docx(file):
    from docx import Document
    doc = Document(file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text
  
# Helper function to extract text from Excel files  
def extract_text_from_excel(file):
    try:
        # Read the Excel file
        df = pd.read_excel(file, sheet_name=None)  # sheet_name=None reads all sheets
        text = ''
        
        # Iterate over all sheets and rows
        for sheet_name, sheet_data in df.items():
            for _, row in sheet_data.iterrows():
                # Convert each cell to a string, handle NaN values
                row_text = ' '.join(str(cell) if pd.notna(cell) else '' for cell in row)
                text += row_text + '\n'

        return text
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Load documents on app start
load_documents()

# Route for Home Page
@app.route('/')
@login_required
def home():
    #return render_template('index.html', username=current_user.id)
    # List all file names in the upload folder
    file_names = os.listdir(app.config['UPLOAD_FOLDER'])
    file_names = [f for f in file_names if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]  # Filter only files
    return render_template('index.html', username=current_user.id, file_names=file_names)

# Route to login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate username and password
        if username in users and users[username]['password'] == password:
            #user = User(id=username)
            user = User(id=username, role=users[username]['role'])
            #user = User(id=username, role='admin')
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials, please try again.', 'danger')

    return render_template('login.html')

# Route for logging out
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Route to ask a question
@app.route('/ask', methods=['POST'])
@login_required
def ask():
    data = request.get_json()
    question = data['question']
    context = "\n\n".join(documents.values())

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{context}\n\n{question}"}]
    )
    answer = response['choices'][0]['message']['content']
    return jsonify({'answer': answer})

# Route to load documents (can be called after user login)
@app.route('/load_documents', methods=['POST'])
@login_required
def load_docs():
    load_documents()
    return jsonify({"message": "Documents loaded successfully", "documents": list(documents.keys())}), 200

# Flask-Login user loader
#@login_manager.user_loader
#def load_user(user_id):
    #return User(user_id)

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        # Make sure we load both id and role from the users dictionary
        return User(user_id, users[user_id]['role'])
    return None

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the 'file' part is in the request
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return 'No selected file'
        
        # If the file is allowed, save it to the upload folder
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(DOCUMENT_FOLDER, filename))  # Save the file
            return f'File successfully uploaded to {DOCUMENT_FOLDER}/{filename}'
    
           # Return the HTML page with the success message (if any)
            return render_template_string('''
            <!doctype html>
            <title>Upload a File</title>
            <h1>Upload a new file</h1>

            {% if success_message %}
                <p style="color: green;">{{ success_message }}</p>
            {% endif %}

            <form method="post" enctype="multipart/form-data">
              <input type="file" name="file">  <!-- Ensure the name is 'file' -->
              <input type="submit" value="Upload">
            </form>
            ''', success_message=success_message)

if __name__ == '__main__':
    app.run(debug=True)
