class DeepResearchApp {
    constructor() {
        this.chatHistory = [];
        this.documents = [];
        this.isThinking = false;
        
        // DOM element bindings
        this.dom = {
            loadingScreen: document.getElementById('loadingScreen'),
            appWrapper: document.getElementById('appWrapper'),
            progressFill: document.getElementById('progressFill'),
            loadingMessage: document.getElementById('loadingMessage'),
            sidebar: document.getElementById('sidebar'),
            documentsList: document.getElementById('documentsList'),
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            chatContainer: document.getElementById('chatContainer'),
            queryInput: document.getElementById('queryInput'),
            sendBtn: document.getElementById('sendBtn'),
            uploadBtn: document.getElementById('uploadBtn'),
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            menuBtn: document.getElementById('menuBtn'),
            closeSidebarBtn: document.getElementById('closeSidebarBtn'),
            exportPdfBtn: document.getElementById('exportPdfBtn'),
        };

        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.checkModelStatus();
        this.addMessageToChat('ai', "Hello! I'm your Deep Research Agent. Please upload some documents to get started.");
    }

    setupEventListeners() {
        this.dom.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.dom.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });
        this.dom.queryInput.addEventListener('input', this.autoResizeTextarea.bind(this));

        // File Upload
        this.dom.uploadBtn.addEventListener('click', () => this.dom.fileInput.click());
        this.dom.uploadArea.addEventListener('click', () => this.dom.fileInput.click());
        this.dom.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));
        
        // Drag and Drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dom.uploadArea.addEventListener(eventName, this.preventDefaults.bind(this), false);
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dom.uploadArea.addEventListener(eventName, () => this.dom.uploadArea.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            this.dom.uploadArea.addEventListener(eventName, () => this.dom.uploadArea.classList.remove('dragover'), false);
        });
        this.dom.uploadArea.addEventListener('drop', (e) => this.handleFileUpload(e.dataTransfer.files), false);
        
        // Sidebar Toggle
        this.dom.menuBtn.addEventListener('click', () => this.dom.sidebar.classList.add('visible'));
        this.dom.closeSidebarBtn.addEventListener('click', () => this.dom.sidebar.classList.remove('visible'));
        
        // Export to PDF
        this.dom.exportPdfBtn.addEventListener('click', this.exportChatToPDF.bind(this));
    }

    preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    autoResizeTextarea() {
        this.dom.queryInput.style.height = 'auto';
        this.dom.queryInput.style.height = `${this.dom.queryInput.scrollHeight}px`;
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            this.dom.loadingMessage.textContent = status.message;
            this.dom.progressFill.style.width = `${status.progress}%`;
            this.dom.statusDot.className = `status-dot ${status.status}`;
            this.dom.statusText.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);

            if (status.status === 'loaded') {
                this.dom.loadingScreen.style.display = 'none';
                this.dom.appWrapper.style.display = 'flex';
                this.loadDocuments();
                return;
            } else if (status.status === 'error') {
                this.dom.loadingMessage.textContent = `Error: ${status.message}`;
                return;
            }
            setTimeout(() => this.checkModelStatus(), 1000);
        } catch (error) {
            console.error('Status check failed:', error);
            setTimeout(() => this.checkModelStatus(), 2000);
        }
    }

    async handleFileUpload(files) {
        if (!files.length) return;
        this.dom.sidebar.classList.remove('visible');
        
        const uploadPromises = Array.from(files).map(file => {
            const formData = new FormData();
            formData.append('file', file);
            return fetch('/api/upload', { method: 'POST', body: formData });
        });
        
        this.addMessageToChat('system', `Uploading ${files.length} document(s)...`);
        
        const responses = await Promise.all(uploadPromises);
        let successCount = 0;
        
        for (const response of responses) {
            const result = await response.json();
            if (result.success) {
                successCount++;
            } else {
                console.error('Upload failed:', result.error);
            }
        }
        
        if (successCount > 0) {
           this.addMessageToChat('system', `Successfully added ${successCount} document(s) to the knowledge base.`);
        }
        this.loadDocuments();
    }
    
    async loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            const result = await response.json();
            this.documents = result.documents || [];
            this.renderDocuments();
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }
    
    renderDocuments() {
        if (this.documents.length === 0) {
            this.dom.documentsList.innerHTML = `<p class="no-docs">No documents uploaded yet.</p>`;
            return;
        }
        this.dom.documentsList.innerHTML = this.documents.map(doc => `
            <div class="document-card" data-id="${doc.id}">
                <div class="document-info">
                    <i class="fas fa-file-alt"></i>
                    <p title="${this.escapeHtml(doc.title)}">${this.escapeHtml(doc.title)}</p>
                </div>
                <button class="btn-icon btn-delete" onclick="app.deleteDocument('${doc.id}', event)">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
        `).join('');
    }

    async deleteDocument(docId, event) {
        event.stopPropagation();
        if (!confirm('Are you sure you want to delete this document?')) return;
        
        try {
            const response = await fetch('/api/delete_document', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ doc_id: docId })
            });
            const result = await response.json();
            if (result.success) {
                this.addMessageToChat('system', `Document removed from knowledge base.`);
                this.loadDocuments();
            } else {
                console.error('Failed to delete document:', result.error);
            }
        } catch (error) {
            console.error('Error deleting document:', error);
        }
    }

    async handleSendMessage() {
        const query = this.dom.queryInput.value.trim();
        if (!query || this.isThinking) return;
        
        if (this.documents.length === 0) {
            this.addMessageToChat('ai', "Please upload a document before asking questions.");
            return;
        }

        this.addMessageToChat('user', query);
        this.dom.queryInput.value = '';
        this.autoResizeTextarea();
        this.setThinking(true);
        
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            
            const result = await response.json();
            this.setThinking(false);

            if (result.error) {
                this.addMessageToChat('ai', `Sorry, an error occurred: ${result.error}`);
            } else {
                this.addMessageToChat('ai', result.answer, result.reasoning_steps);
            }

        } catch (error) {
            this.setThinking(false);
            this.addMessageToChat('ai', 'Sorry, I was unable to process your request.');
            console.error('Query failed:', error);
        }
    }

    setThinking(isThinking) {
        this.isThinking = isThinking;
        this.dom.sendBtn.disabled = isThinking;
        
        const thinkingBubble = document.getElementById('thinking-bubble');
        if (isThinking && !thinkingBubble) {
            const messageEl = document.createElement('div');
            messageEl.className = 'chat-message ai';
            messageEl.id = 'thinking-bubble';
            messageEl.innerHTML = `
                <div class="message-bubble">
                    <div class="thinking-bubble">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>`;
            this.dom.chatContainer.appendChild(messageEl);
            this.scrollToBottom();
        } else if (!isThinking && thinkingBubble) {
            thinkingBubble.remove();
        }
    }

    addMessageToChat(sender, text, reasoning = null) {
        const message = { sender, text, reasoning, timestamp: new Date() };
        this.chatHistory.push(message);

        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}`;
        
        let contentHTML = `<div class="message-bubble ${sender}">${this.formatText(text)}</div>`;
        
        if (reasoning && reasoning.length > 0) {
            const reasoningId = `reasoning-${Date.now()}`;
            contentHTML += `
                <div class="message-footer">
                    <span class="reasoning-toggle" onclick="app.toggleReasoning('${reasoningId}')">
                        <i class="fas fa-cogs"></i> Show Reasoning
                    </span>
                    <div class="reasoning-content" id="${reasoningId}">
                        <ol>${reasoning.map(step => `<li>${this.escapeHtml(step)}</li>`).join('')}</ol>
                    </div>
                </div>`;
        }
        
        messageEl.innerHTML = contentHTML;
        this.dom.chatContainer.appendChild(messageEl);
        this.scrollToBottom();
    }

    toggleReasoning(id) {
        const el = document.getElementById(id);
        const toggle = el.previousElementSibling;
        el.classList.toggle('visible');
        if (el.classList.contains('visible')) {
            toggle.innerHTML = `<i class="fas fa-cogs"></i> Hide Reasoning`;
        } else {
            toggle.innerHTML = `<i class="fas fa-cogs"></i> Show Reasoning`;
        }
    }

    async exportChatToPDF() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF({
            orientation: 'p',
            unit: 'mm',
            format: 'a4'
        });

        const margin = 15;
        const pageWidth = doc.internal.pageSize.getWidth();
        const usableWidth = pageWidth - (2 * margin);
        let cursorY = margin;

        // Title
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(20);
        doc.text('Deep Research - Chat Export', pageWidth / 2, cursorY, { align: 'center' });
        cursorY += 15;

        // Metadata
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.text(`Exported on: ${new Date().toLocaleString()}`, margin, cursorY);
        cursorY += 10;
        doc.setLineWidth(0.2);
        doc.line(margin, cursorY, pageWidth - margin, cursorY);
        cursorY += 10;
        
        for (const message of this.chatHistory) {
            if (cursorY > 270) { // New page check
                doc.addPage();
                cursorY = margin;
            }

            doc.setFontSize(12);
            doc.setFont('helvetica', message.sender === 'user' ? 'bold' : 'bold');
            const senderText = message.sender.charAt(0).toUpperCase() + message.sender.slice(1);
            doc.text(`${senderText}:`, margin, cursorY);
            
            doc.setFont('helvetica', 'normal');
            const splitText = doc.splitTextToSize(message.text, usableWidth - 5);
            doc.text(splitText, margin + 5, cursorY + 5);
            
            cursorY += (splitText.length * 5) + 10;
        }

        doc.save(`research-chat-${new Date().toISOString().split('T')[0]}.pdf`);
    }

    scrollToBottom() {
        this.dom.chatContainer.scrollTop = this.dom.chatContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatText(text) {
        return this.escapeHtml(text)
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DeepResearchApp();
    window.app = app; // For debugging
});
