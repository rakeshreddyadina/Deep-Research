from flask import Flask, render_template, request, jsonify
import os
import logging
from werkzeug.utils import secure_filename
import threading
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for Word document processing

from models.embedding_manager import EmbeddingManager
from models.document_storage import DocumentStorage
from models.query_processor import QueryProcessor
from models.answer_generator import AnswerGenerator

app = Flask(__name__)
# It's highly recommended to set this key from an environment variable for security
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-very-secret-dev-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
embedding_manager = None
document_storage = None
query_processor = None
answer_generator = None
model_loading_status = {"status": "not_started", "progress": 0, "message": ""}
model_init_thread = None


def initialize_models():
    """Initializes all AI models and necessary components."""
    global embedding_manager, document_storage, query_processor, answer_generator, model_loading_status
    try:
        model_loading_status.update({"status": "loading", "message": "Initializing...", "progress": 10})

        document_storage = DocumentStorage()
        logger.info("Document storage initialized")
        model_loading_status.update({"message": "Document storage ready", "progress": 20})

        query_processor = QueryProcessor()
        logger.info("Query processor initialized")
        model_loading_status.update({"message": "Query processor ready", "progress": 30})

        embedding_manager = EmbeddingManager()
        logger.info("Embedding manager initialized")
        model_loading_status.update({"message": "Embedding model loaded", "progress": 60})

        answer_generator = AnswerGenerator()
        logger.info("Answer generator initialized")
        model_loading_status.update({"message": "Generative model loaded", "progress": 90})

        model_loading_status.update({"status": "loaded", "message": "Ready", "progress": 100})
        logger.info("All models initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing models: {e}", exc_info=True)
        model_loading_status.update({"status": "error", "message": f"Initialization failed: {e}"})


# --- Text Extraction Utilities ---
def extract_text_from_pdf(filepath):
    with fitz.open(filepath) as doc:
        return "".join(page.get_text() for page in doc)


def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)


def add_document_to_storage(title, content, source):
    embedding = embedding_manager.encode_document(content, title)
    return document_storage.add_document(title=title, content=content, source=source, embedding=embedding)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main application and starts model initialization on first load."""
    global model_init_thread
    if model_init_thread is None:
        model_init_thread = threading.Thread(target=initialize_models, daemon=True)
        model_init_thread.start()
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    return jsonify(model_loading_status)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handles uploads of .txt, .pdf, and .docx files."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ['.txt', '.pdf', '.docx']:
        return jsonify({'error': 'File type not supported'}), 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        content = ""
        if file_ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext == '.pdf':
            content = extract_text_from_pdf(filepath)
        elif file_ext == '.docx':
            content = extract_text_from_docx(filepath)

        if not content.strip():
            os.remove(filepath)
            return jsonify({'error': 'File contains no readable text'}), 400

        doc_id = add_document_to_storage(filename, content, filepath)
        return jsonify({'success': True, 'filename': filename, 'doc_id': doc_id})
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        return jsonify({'error': f'Could not process file: {e}'}), 500


@app.route('/api/documents')
def get_documents():
    if not document_storage: return jsonify({'documents': []})
    docs = document_storage.get_all_documents()
    return jsonify({'documents': [doc.to_dict() for doc in docs]})


@app.route('/api/delete_document', methods=['POST'])
def delete_document():
    doc_id = request.get_json().get('doc_id')
    if not doc_id: return jsonify({'error': 'Document ID is required'}), 400
    if document_storage.delete_document(doc_id):
        return jsonify({'success': True})
    return jsonify({'error': 'Document not found'}), 404


@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({'error': 'Query is required'}), 400

    try:
        query_embedding = embedding_manager.encode_query(query)
        retrieved_docs = document_storage.search_by_similarity(query_embedding, top_k=3)
        documents = [doc for doc, score in retrieved_docs]

        reasoning_steps = query_processor.generate_reasoning_steps(query, retrieved_docs)
        final_answer, confidence, sources = answer_generator.generate_answer(query, documents, reasoning_steps)

        return jsonify({
            'answer': final_answer,
            'reasoning_steps': reasoning_steps,
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('storage', exist_ok=True)
    os.makedirs('models_cache', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
