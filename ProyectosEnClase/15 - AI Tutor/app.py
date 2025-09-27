import streamlit as st
import PyPDF2
from io import BytesIO
import openai
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, TypedDict, Annotated
import re
import json
import logging
import io
import sys
import traceback
import requests
from urllib.parse import quote_plus
import base64

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import formula/math libraries
try:
    import sympy as sp
    from sympy import latex, sympify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Import web search libraries
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    st.error("ChromaDB not installed. Please run: pip install chromadb")

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Study Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with proper contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #1a1a1a !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
        color: #1a1a1a !important;
    }
    .assistant-message {
        background-color: #f9f9f9;
        border-left: 4px solid #ff7f0e;
        color: #1a1a1a !important;
    }
    .pdf-content {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 5px;
        max-height: 300px;
        overflow-y: auto;
        color: #1a1a1a !important;
    }
    /* Force text color in all message elements */
    .chat-message * {
        color: #1a1a1a !important;
    }
    /* Streamlit markdown overrides */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1a1a1a !important;
    }
    .stMarkdown p {
        color: #1a1a1a !important;
    }
    .stMarkdown code {
        background-color: #f0f0f0 !important;
        color: #1a1a1a !important;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .stMarkdown pre {
        background-color: #f8f8f8 !important;
        color: #1a1a1a !important;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "pdf_documents" not in st.session_state:
    st.session_state.pdf_documents = []
if "tutor_agent" not in st.session_state:
    st.session_state.tutor_agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4"
if "response_length" not in st.session_state:
    st.session_state.response_length = "medium"
if "code_outputs" not in st.session_state:
    st.session_state.code_outputs = []
if "plot_outputs" not in st.session_state:
    st.session_state.plot_outputs = []

def extract_pdf_text(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        error_msg = f"Error extracting PDF text: {str(e)}"
        st.session_state.last_error = error_msg
        st.error(error_msg)
        return ""

def initialize_chroma_vectorstore(api_key: str):
    """Initialize ChromaDB with OpenAI embeddings."""
    try:
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not available")

        # Initialize OpenAI embeddings with text-embedding-3-large model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )

        # Initialize Chroma client (in-memory)
        chroma_client = chromadb.Client(Settings(
            #persist_directory=None,  # In-memory
            is_persistent=False
        ))

        # Create or get collection
        collection_name = "pdf_documents"
        try:
            collection = chroma_client.get_collection(collection_name)
        except:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "PDF document chunks with embeddings"}
            )

        st.session_state.chroma_client = chroma_client
        st.session_state.collection = collection
        st.session_state.embeddings = embeddings

        return chroma_client, collection, embeddings

    except Exception as e:
        error_msg = f"Error initializing Chroma vectorstore: {str(e)}"
        st.session_state.last_error = error_msg
        st.error(error_msg)
        return None, None, None

def add_documents_to_vectorstore(documents: List[Document], embeddings, collection):
    """Add Document objects to Chroma vectorstore with embeddings and metadata."""
    try:
        if not documents or not embeddings or not collection:
            return False

        # Extract text content for embedding
        texts = [doc.page_content for doc in documents]

        # Generate embeddings for document chunks
        with st.spinner("Creating embeddings for document chunks..."):
            chunk_embeddings = embeddings.embed_documents(texts)

        # Prepare documents for Chroma with enhanced metadata
        ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]
        metadatas = []

        for i, doc in enumerate(documents):
            # Combine existing metadata with additional info
            metadata = doc.metadata.copy()
            metadata.update({
                "text_length": len(doc.page_content),
                "embedding_model": "text-embedding-3-large",
                "added_to_vectorstore_at": str(pd.Timestamp.now()) if 'pd' in globals() else str(i)
            })
            metadatas.append(metadata)

        # Add to collection
        collection.add(
            embeddings=chunk_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        return True

    except Exception as e:
        error_msg = f"Error adding documents to vectorstore: {str(e)}"
        st.session_state.last_error = error_msg
        st.error(error_msg)
        return False

def vector_search(query: str, embeddings, collection, top_k: int = 3) -> List[str]:
    """Search for relevant chunks using vector similarity."""
    try:
        if not query or not embeddings or not collection:
            return []

        # Generate embedding for query
        query_embedding = embeddings.embed_query(query)

        # Search in collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Extract documents from results
        if results and results.get('documents') and results['documents'][0]:
            return results['documents'][0]

        return []

    except Exception as e:
        error_msg = f"Error performing vector search: {str(e)}"
        st.session_state.last_error = error_msg
        st.error(error_msg)
        return []

def display_code_tool(code: str, language: str = "python", description: str = "") -> str:
    """Display code examples with syntax highlighting for multiple languages."""
    try:
        # Detect language from context if not specified
        if language == "python" and any(keyword in code.lower() for keyword in ["def ", "import ", "print(", "if __name__"]):
            language = "python"
        elif "function" in code.lower() and "{" in code:
            language = "javascript"
        elif "public class" in code or "public static void main" in code:
            language = "java"
        elif "#include" in code or "int main(" in code:
            language = "cpp"
        elif "def " in code and "end" in code:
            language = "ruby"
        elif "fn " in code and "let " in code:
            language = "rust"
        elif "package " in code and "func " in code:
            language = "go"

        result = f"**Code Example ({language.title()}):**\n```{language}\n{code}\n```\n"
        if description:
            result += f"\n**Description:** {description}\n"

        # Store in session state for display
        st.session_state.code_outputs.append({
            "code": code,
            "language": language,
            "description": description
        })

        return result

    except Exception as e:
        error_msg = f"Code display error: {str(e)}"
        st.session_state.last_error = error_msg
        return f"**Error displaying code:**\n```\n{error_msg}\n```"

def create_plot_tool(plot_type: str, data: str, title: str = "", **kwargs) -> str:
    """Create and display plots using matplotlib or plotly."""
    try:
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            return "**Error:** No plotting libraries available. Please install matplotlib or plotly."

        # Parse data (expecting simple format like "x=[1,2,3],y=[4,5,6]")
        plot_data = {}
        if data:
            try:
                # Simple parsing for basic data formats
                exec(data, {}, plot_data)
            except:
                return f"**Error:** Could not parse plot data: {data}"

        fig = None

        if PLOTLY_AVAILABLE and plot_type in ["line", "scatter", "bar", "histogram"]:
            # Use Plotly for interactive plots
            if plot_type == "line" and "x" in plot_data and "y" in plot_data:
                fig = go.Figure(data=go.Scatter(x=plot_data["x"], y=plot_data["y"], mode='lines+markers'))
            elif plot_type == "scatter" and "x" in plot_data and "y" in plot_data:
                fig = go.Figure(data=go.Scatter(x=plot_data["x"], y=plot_data["y"], mode='markers'))
            elif plot_type == "bar" and "x" in plot_data and "y" in plot_data:
                fig = go.Figure(data=go.Bar(x=plot_data["x"], y=plot_data["y"]))
            elif plot_type == "histogram" and "values" in plot_data:
                fig = go.Figure(data=go.Histogram(x=plot_data["values"]))

            if fig:
                if title:
                    fig.update_layout(title=title)

                # Store in session state for display
                st.session_state.plot_outputs.append({
                    "type": "plotly",
                    "figure": fig,
                    "title": title
                })

                return f"**📊 Plot Created:** {title or plot_type.title() + ' Plot'}\n*Interactive plot will be displayed below the chat.*"

        elif MATPLOTLIB_AVAILABLE:
            # Use Matplotlib as fallback
            plt.figure(figsize=(8, 6))

            if plot_type == "line" and "x" in plot_data and "y" in plot_data:
                plt.plot(plot_data["x"], plot_data["y"], marker='o')
            elif plot_type == "scatter" and "x" in plot_data and "y" in plot_data:
                plt.scatter(plot_data["x"], plot_data["y"])
            elif plot_type == "bar" and "x" in plot_data and "y" in plot_data:
                plt.bar(plot_data["x"], plot_data["y"])
            elif plot_type == "histogram" and "values" in plot_data:
                plt.hist(plot_data["values"], bins=kwargs.get("bins", 20))
            else:
                plt.text(0.5, 0.5, f"Plot type '{plot_type}' with data format not supported",
                        ha='center', va='center', transform=plt.gca().transAxes)

            if title:
                plt.title(title)
            plt.grid(True, alpha=0.3)

            # Save plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # Store in session state for display
            st.session_state.plot_outputs.append({
                "type": "matplotlib",
                "data": plot_data_b64,
                "title": title
            })

            return f"**📊 Plot Created:** {title or plot_type.title() + ' Plot'}\n*Plot will be displayed below the chat.*"

        return f"**Error:** Unable to create {plot_type} plot with provided data."

    except Exception as e:
        error_msg = f"Plotting error: {str(e)}"
        st.session_state.last_error = error_msg
        return f"**Error creating plot:**\n```\n{error_msg}\n```"

def display_formula_tool(formula: str, description: str = "") -> str:
    """Display mathematical formulas using LaTeX rendering."""
    try:
        if not SYMPY_AVAILABLE:
            return f"**Formula:** {formula}\n{description}\n\n*Note: Install sympy for enhanced formula rendering.*"

        # Try to parse and render the formula
        try:
            # Parse the formula with sympy
            expr = sympify(formula)
            latex_formula = latex(expr)

            # Store for display
            formula_data = {
                "formula": formula,
                "latex": latex_formula,
                "description": description,
                "rendered": str(expr)
            }

            # Display using Streamlit's LaTeX support
            result = f"**Formula:** \n$$\n{latex_formula}\n$$\n"
            if description:
                result += f"\n**Description:** {description}\n"

            return result

        except:
            # Fallback for non-sympy expressions
            result = f"**Formula:** \n$$\n{formula}\n$$\n"
            if description:
                result += f"\n**Description:** {description}\n"
            return result

    except Exception as e:
        error_msg = f"Formula rendering error: {str(e)}"
        st.session_state.last_error = error_msg
        return f"**Error displaying formula:**\n{formula}\n{description}"

def web_search_tool(query: str, num_results: int = 5) -> str:
    """Intelligent web search using DuckDuckGo."""
    try:
        if not DUCKDUCKGO_AVAILABLE:
            # Fallback to basic search suggestion
            search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
            return f"**🔍 Search Query:** {query}\n\n*Web search libraries not available. You can search manually at: {search_url}*"

        # Perform search with DuckDuckGo
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return f"**🔍 Search Query:** {query}\n\nNo results found."

        # Format results
        search_result = f"**🔍 Web Search Results for:** {query}\n\n"

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', '')

            # Truncate long descriptions
            if len(body) > 200:
                body = body[:200] + "..."

            search_result += f"**{i}. {title}**\n"
            search_result += f"{body}\n"
            search_result += f"🔗 {href}\n\n"

        # Store search results for potential follow-up
        return search_result

    except Exception as e:
        error_msg = f"Web search error: {str(e)}"
        st.session_state.last_error = error_msg
        return f"**Error performing web search:**\n```\n{error_msg}\n```\n\nTry rephrasing your search query."

def split_text_into_documents(text: str, source: str = "uploaded_pdf") -> tuple[List[Document], List[str]]:
    """Split text into Document objects with metadata using RecursiveCharacterTextSplitter."""
    if not text:
        return [], []

    # Create a Document object with metadata
    doc = Document(
        page_content=text,
        metadata={
            "source": source,
            "total_length": len(text),
            "doc_type": "pdf",
            "processed_at": str(st.session_state.get("last_upload_time", "unknown"))
        }
    )

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,  # Size of each chunk
        chunk_overlap=200,  # Overlap between chunks to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical splitting
    )

    # Split the document into smaller documents with metadata
    split_docs = text_splitter.split_documents([doc])

    # Update metadata for each chunk
    for i, split_doc in enumerate(split_docs):
        split_doc.metadata.update({
            "chunk_index": i,
            "chunk_id": f"{source}_chunk_{i}",
            "chunk_length": len(split_doc.page_content),
            "total_chunks": len(split_docs)
        })

    # Extract text chunks for backward compatibility
    text_chunks = [doc.page_content for doc in split_docs]

    return split_docs, text_chunks

def find_relevant_chunks(chunks: List[str], query: str, max_chunks: int = 3) -> List[str]:
    """Find the most relevant chunks for a given query using vector search when available."""
    if not chunks or not query:
        return []

    # Try vector search first if Chroma is available and initialized
    if (st.session_state.embeddings and st.session_state.collection):
        try:
            vector_results = vector_search(query, st.session_state.embeddings, st.session_state.collection, max_chunks)
            if vector_results:
                return vector_results
        except Exception as e:
            st.session_state.last_error = f"Vector search failed, falling back to keyword search: {str(e)}"
            st.warning("Vector search failed, using keyword search instead.")

    # Fallback to keyword-based search
    query_lower = query.lower()
    scored_chunks = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Simple scoring based on keyword matches
        score = 0
        query_words = query_lower.split()

        for word in query_words:
            if len(word) > 2:  # Ignore very short words
                score += chunk_lower.count(word)

        # Bonus for exact phrase matches
        if query_lower in chunk_lower:
            score += 10

        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:max_chunks]]

def create_study_tools(pdf_content: str, pdf_chunks: List[str]) -> List:
    """Create tools for the LangGraph ReAct agent based on PDF content."""

    @tool
    def search_content(query: str) -> str:
        """Search for specific information, concepts, or keywords in the uploaded PDF document using intelligent chunking and vector database. Use this when the student asks about specific topics or concepts."""
        if not pdf_chunks:
            return "No PDF content available. Please upload a PDF first."

        # Find relevant chunks using vector search when available, fallback to keyword search
        relevant_chunks = find_relevant_chunks(pdf_chunks, query, max_chunks=3)

        if relevant_chunks:
            result = f"Based on the document content, here's what I found about '{query}':\n\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                # Truncate very long chunks for readability
                display_chunk = chunk[:600] + "..." if len(chunk) > 600 else chunk
                result += f"**Section {i}:**\n{display_chunk}\n\n"

            result += "Would you like me to explain any of these concepts in more detail or create a quiz to test your understanding?"
            return result
        else:
            return f"I couldn't find specific information about '{query}' in the document. Try using different keywords, or ask me to provide a document overview to see what topics are covered."

    @tool
    def create_quiz(topic: str) -> str:
        """
        Generates a dynamic quiz with questions, answers, and explanations based on a specific topic from the PDF content.
        Use this when the student wants to test their knowledge autonomously.
        """
        if not pdf_chunks:
            return "No PDF content available. Please upload a PDF first."

        # 1. Find relevant content for the quiz topic
        relevant_chunks = find_relevant_chunks(pdf_chunks, topic, max_chunks=4)
        if not relevant_chunks:
            return f"I couldn't find enough information about '{topic}' to generate a quiz. Please try another topic."

        context = "\n---\n".join(relevant_chunks)

        # 2. Create a prompt for the LLM to generate the quiz
        # This assumes an 'llm' object is available in the scope to process the request.
        try:
            prompt = f"""
            Based ONLY on the following text from a document, act as a helpful study assistant.
            Generate a short quiz with 2-3 questions about '{topic}'.

            For each question, you MUST provide:
            1. A clear, concise question.
            2. A detailed answer to the question.
            3. A brief 'Explanation from the text' that justifies the answer by quoting or summarizing the provided source text.

            Format the entire output in Markdown. Start with a title for the quiz.

            Here is the source text:
            ---
            {context}
            ---
            """

            # NOTE: The following line is a placeholder.
            # You must replace it with the actual call to your language model.
            # For example, if you are using a LangChain LLM object named 'llm':
            # response = llm.invoke(prompt)
            # return response.content

            # This placeholder simulates the expected output structure.
            quiz_simulation = (
                f"📝 **Quiz on {topic}**\n\n"
                "Here are a few questions based on the document content:\n\n"
                "---\n\n"
                "**Question 1:** (This is where the AI-generated question about the topic would appear).\n\n"
                "**Answer:** (This is where the AI-generated answer would appear).\n\n"
                "**Explanation from the text:** (The AI would provide a justification based on the source text provided).\n\n"
                "---\n\n"
                "**Question 2:** (This is where the second AI-generated question would appear).\n\n"
                "**Answer:** (This is where the second AI-generated answer would appear).\n\n"
                "**Explanation from the text:** (The AI would provide another justification based on the source text).\n\n"
            )
            return quiz_simulation

        except Exception as e:
            return f"An error occurred while generating the quiz: {e}"

    @tool
    def summarize_section(section: str) -> str:
        """Provide a detailed summary of a specific section or concept from the PDF using actual document content. Use this when the student wants an overview of a topic."""
        if not pdf_chunks:
            return "No PDF content available. Please upload a PDF first."

        # Find relevant chunks for the section using vector search
        relevant_chunks = find_relevant_chunks(pdf_chunks, section, max_chunks=4)

        if relevant_chunks:
            combined_content = "\n".join(relevant_chunks)

            result = f"📋 **Summary of '{section}' from your document:**\n\n"

            # Provide the most relevant content with better formatting
            if len(combined_content) > 1200:
                result += combined_content[:1200] + "...\n\n"
            else:
                result += combined_content + "\n\n"

            # Extract key points from the content
            sentences = re.split(r'[.!?]+', combined_content)
            key_sentences = [
                sentence.strip() for sentence in sentences
                if len(sentence.strip()) > 40 and (
                    section.lower() in sentence.lower() or
                    any(word in sentence.lower() for word in section.lower().split())
                )
            ]

            if key_sentences:
                result += "**Key Points:**\n"
                for i, sentence in enumerate(key_sentences[:4], 1):
                    result += f"{i}. {sentence}.\n"
                result += "\n"

            result += "Would you like me to create a quiz on this topic or search for more specific information?"
            return result
        else:
            return f"I couldn't find specific information about '{section}' in the document. The document might use different terminology. Try searching with related keywords or ask me for a document overview to see what topics are covered."

    @tool
    def get_document_overview() -> str:
        """Provide an overview of the entire document structure, length, and main topics using actual document analysis. Use this when the student wants to understand what the document covers overall."""
        if not pdf_chunks:
            return "No PDF content available. Please upload a PDF first."

        # Analyze document structure using actual content
        result = "📚 **Document Overview:**\n\n"
        result += f"**Document Statistics:**\n"
        result += f"- Total sections/chunks: {len(pdf_chunks)}\n"
        result += f"- Total characters: {len(pdf_content):,}\n"
        result += f"- Estimated reading time: {len(pdf_content) // 1000} minutes\n\n"

        # Extract potential topics and themes from first and random chunks
        sample_chunks = pdf_chunks[:3] + (pdf_chunks[len(pdf_chunks)//2:len(pdf_chunks)//2+2] if len(pdf_chunks) > 5 else [])

        # Find common terms and concepts
        all_text = " ".join(sample_chunks).lower()
        words = re.findall(r'\b\w{4,}\b', all_text)
        word_freq = {}
        for word in words:
            if word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'your', 'more', 'some', 'what', 'when', 'where', 'would', 'could', 'should']:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get most frequent meaningful terms
        common_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        if common_terms:
            result += "**Main Topics/Concepts (based on frequency):**\n"
            for i, (term, freq) in enumerate(common_terms, 1):
                if freq > 2:  # Only show terms that appear multiple times
                    result += f"{i}. {term.title()} (mentioned {freq} times)\n"
            result += "\n"

        # Extract content previews
        result += "**Content Preview:**\n"
        for i, chunk in enumerate(sample_chunks[:3], 1):
            lines = chunk.split('\n')
            significant_line = next((line.strip() for line in lines if len(line.strip()) > 30), "")
            if significant_line:
                preview = significant_line[:120] + "..." if len(significant_line) > 120 else significant_line
                result += f"{i}. {preview}\n"

        result += "\n**How to Study:**\n"
        result += "- Ask me about specific topics using 'search_content'\n"
        result += "- Request summaries of particular concepts\n"
        result += "- Generate quizzes to test your understanding\n"
        result += "- Ask for code examples, plots, or formulas related to the content\n"

        return result

    @tool
    def display_code(query: str) -> str:
        """Display code examples with syntax highlighting in multiple programming languages, contextually relevant to the query or document content. Use this when students need to see code examples."""

        # First, check if there's code-related content in the document
        code_context = ""
        if pdf_chunks:
            # Search for programming-related content in the document
            programming_chunks = []
            for chunk in pdf_chunks:
                if any(term in chunk.lower() for term in ['code', 'program', 'function', 'algorithm', 'implementation', 'syntax', 'variable', 'loop', 'class', 'method']):
                    programming_chunks.append(chunk)

            if programming_chunks:
                # Use document context to inform the code example
                relevant_context = programming_chunks[0][:200] + "..." if len(programming_chunks[0]) > 200 else programming_chunks[0]
                code_context = f"\n**Related to your document content:**\n{relevant_context}\n\n"

        # Extract language from query
        language = "python"  # default
        query_lower = query.lower()

        if any(lang in query_lower for lang in ["javascript", "js", "node"]):
            language = "javascript"
        elif any(lang in query_lower for lang in ["java"]) and "javascript" not in query_lower:
            language = "java"
        elif any(lang in query_lower for lang in ["c++", "cpp", "c plus"]):
            language = "cpp"
        elif any(lang in query_lower for lang in ["ruby", "rb"]):
            language = "ruby"
        elif any(lang in query_lower for lang in ["rust", "rs"]):
            language = "rust"
        elif any(lang in query_lower for lang in ["go", "golang"]):
            language = "go"
        elif any(lang in query_lower for lang in ["sql", "database", "select", "insert"]):
            language = "sql"
        elif any(lang in query_lower for lang in ["html", "web", "markup"]):
            language = "html"
        elif any(lang in query_lower for lang in ["css", "style"]):
            language = "css"

        # Generate code example based on document context and query
        if "data" in query_lower or "analysis" in query_lower or (pdf_content and any(term in pdf_content.lower() for term in ["data", "analysis", "statistics"])):
            code = f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze data
data = pd.read_csv('your_data.csv')
print(f"Dataset shape: {{data.shape}}")
print(f"Column names: {{data.columns.tolist()}}")

# Basic statistics
print(data.describe())

# Data visualization
plt.figure(figsize=(10, 6))
data.plot(kind='hist', alpha=0.7)
plt.title('Data Distribution')
plt.show()'''
            description = f"Data analysis example in {language.title()} based on your document content"

        elif "algorithm" in query_lower or "search" in query_lower:
            code = f'''def binary_search(arr, target):
    """
    Binary search algorithm implementation
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example usage
numbers = [1, 3, 5, 7, 9, 11, 13, 15]
result = binary_search(numbers, 7)
print(f"Found at index: {{result}}")'''
            description = f"Binary search algorithm in {language.title()}"

        else:
            # Generate a basic example
            code = f'''# Basic {language.title()} example
def main():
    """
    Example function demonstrating basic concepts
    """
    message = "Hello, World!"
    print(message)

    # Process data
    numbers = [1, 2, 3, 4, 5]
    squared = [x**2 for x in numbers]

    print(f"Original: {{numbers}}")
    print(f"Squared: {{squared}}")

if __name__ == "__main__":
    main()'''
            description = f"Basic {language.title()} example"

        result = code_context + display_code_tool(code, language, description)
        return result

    @tool
    def create_plot(plot_type: str, data: str, title: str = "") -> str:
        """Create and display plots for data visualization. Use format: plot_type (line/scatter/bar/histogram), data (e.g., 'x=[1,2,3],y=[4,5,6]'), title. Use this when students need visual representations of data."""
        return create_plot_tool(plot_type, data, title)

    @tool
    def display_formula(formula: str, description: str = "") -> str:
        """Display mathematical formulas with LaTeX rendering. Pass the formula as a string (e.g., 'x^2 + y^2 = z^2') and optional description. Use this when explaining mathematical concepts."""
        return display_formula_tool(formula, description)

    @tool
    def web_search(query: str) -> str:
        """Search the internet for additional information, current events, or topics not covered in the PDF. Use this when students ask about topics beyond the document or need updated information."""
        return web_search_tool(query, num_results=5)

    return [search_content, create_quiz, summarize_section, get_document_overview, display_code, create_plot, display_formula, web_search]

def create_react_tutor(api_key: str, pdf_content: str, pdf_chunks: List[str]):
    """Create a LangGraph ReAct agent-based tutor that autonomously selects and uses tools."""

    # Get response length setting
    response_length_settings = {
        "short": {"max_tokens": 150, "temperature": 0.6},
        "medium": {"max_tokens": 400, "temperature": 0.7},
        "long": {"max_tokens": 800, "temperature": 0.7},
        "detailed": {"max_tokens": 1200, "temperature": 0.8},
        "extensive": {"max_tokens": 4096, "temperature": 0.9},
    }

    length_setting = response_length_settings.get(st.session_state.response_length, response_length_settings["medium"])

    # Initialize the LLM with selected model and settings
    llm = ChatOpenAI(
        model=st.session_state.selected_model,
        temperature=length_setting["temperature"],
        max_tokens=length_setting["max_tokens"],
        openai_api_key=api_key
    )

    # Create tools for the agent
    tools = create_study_tools(pdf_content, pdf_chunks)

    # Create system message for the ReAct agent
    system_message = """You are an intelligent study tutor helping students learn from their uploaded PDF documents.
    You have access to tools that can search and analyze the document content, but you should also use your general knowledge to provide comprehensive educational responses.

    IMPORTANT GUIDELINES:
    1. For basic questions or general explanations, you can answer directly using your knowledge
    2. Use tools when you need specific information from the student's document
    3. Combine tool results with your expertise to provide comprehensive answers
    4. Always be encouraging, educational, and thorough

    Available tools:
    - search_content: Search the student's PDF document for specific information
    - create_quiz: Generate quizzes based on document content
    - summarize_section: Summarize specific sections from the document
    - get_document_overview: Get an overview of the entire document
    - display_code: Show relevant code examples
    - create_plot: Create data visualizations
    - display_formula: Display mathematical formulas
    - web_search: Search for additional current information

    WHEN TO USE TOOLS:
    - Use search_content when the student asks about specific content in their document
    - Use create_quiz when they want to test their knowledge of document topics
    - Use display_code when they need programming examples (but only if relevant)
    - Use other tools as specifically requested

    WHEN NOT TO USE TOOLS:
    - For general questions that don't require document-specific information
    - For basic explanations of concepts
    - For simple conversations or greetings
    - When you can provide a complete answer with your existing knowledge

    Remember: You're a knowledgeable tutor who can answer many questions directly, and you have tools available for document-specific tasks."""

    # Create the LangGraph ReAct agent with tools
    agent_executor = create_react_agent(
        model=llm,
        tools=tools
    )

    def react_tutor_response(query: str, chat_history: List[Dict]) -> Dict:
        """Generate response using LangGraph ReAct agent."""
        try:
            # Format chat history for context
            formatted_history = []

            # Add system message as the first message
            formatted_history.append(AIMessage(content=system_message))

            # Add recent chat history
            for msg in chat_history[-4:]:  # Last 4 messages for context
                if msg['role'] == 'user':
                    formatted_history.append(HumanMessage(content=msg['content']))
                else:
                    formatted_history.append(AIMessage(content=msg['content']))

            # Add current query
            formatted_history.append(HumanMessage(content=query))

            # Invoke the LangGraph ReAct agent
            response = agent_executor.invoke({
                "messages": formatted_history
            })

            # Extract the response from the agent's output
            if response and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_text = last_message.content
                else:
                    response_text = str(last_message)
            else:
                response_text = "I apologize, but I couldn't process your request properly."

            return {
                "result": response_text,
                "tool_used": "langgraph_react_agent"
            }

        except Exception as e:
            error_msg = str(e)
            st.session_state.last_error = error_msg

            # Provide user-friendly error messages
            if "api" in error_msg.lower() or "openai" in error_msg.lower():
                result = "I'm having trouble connecting to the AI service. Please check your API key and try again."
            elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                result = "I'm currently experiencing high demand. Please try again in a moment."
            elif "parsing" in error_msg.lower():
                result = "I had trouble understanding the request. Could you try rephrasing your question?"
            elif "tool" in error_msg.lower():
                result = "I encountered an issue using one of my tools. Let me try to help you in a different way."
            else:
                result = f"I encountered an issue processing your request. Please try asking in a different way, or ask me to help with specific topics like summaries, quizzes, code examples, or document searches."

            return {
                "result": result,
                "tool_used": "error_fallback"
            }

    return react_tutor_response

# Main UI
st.markdown('<h1 class="main-header">📚 PDF Study Tutor & Coach</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable the tutor agent")

    # Model Selection
    st.subheader("🤖 AI Configuration")
    available_models = [
    # 2025 releases
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",

    # 2024 advanced reasoning + turbo
    "o3",
    "o3-mini",
    "o1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",

    # 2023 GPT-4 stable + 32k context
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-4-32k-0613",
    "gpt-4-32k-0314",

    # 2023 GPT-3.5 refresh
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k-0613",

    # Base 2022–2023
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
]

    selected_model = st.selectbox(
        "Select AI Model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        help="Choose the AI model for responses. GPT-4 models provide higher quality but may be slower."
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Reset tutor agent to use new model
        if st.session_state.tutor_agent and api_key:
            st.session_state.tutor_agent = None
            st.info("Model changed. Please process your PDF again to update the tutor.")

    # Response Length Control
    response_lengths = {
        "short": "Short (150 tokens) - Quick answers",
        "medium": "Medium (400 tokens) - Balanced responses",
        "long": "Long (800 tokens) - Detailed explanations",
        "detailed": "Detailed (1200 tokens) - Comprehensive responses",
        "extensive": "Use max explainability Ultra Comprehensive responses"        
    }

    selected_length = st.selectbox(
        "Response Length",
        list(response_lengths.keys()),
        index=list(response_lengths.keys()).index(st.session_state.response_length),
        format_func=lambda x: response_lengths[x],
        help="Control the length and detail level of AI responses."
    )

    if selected_length != st.session_state.response_length:
        st.session_state.response_length = selected_length
        # Reset tutor agent to use new settings
        if st.session_state.tutor_agent and api_key:
            st.session_state.tutor_agent = None
            st.info("Response length changed. Please process your PDF again to update the tutor.")
    
    # PDF Upload
    st.header("📄 Upload Study Material")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload your study material (textbook, notes, etc.)"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_pdf_text(uploaded_file)
                if pdf_text:
                    st.session_state.pdf_content = pdf_text
                    st.session_state.last_upload_time = pd.Timestamp.now() if 'pd' in globals() else "now"

                    # Split text into Document objects with metadata
                    with st.spinner("Processing text into intelligent chunks with metadata..."):
                        filename = uploaded_file.name if hasattr(uploaded_file, 'name') else "uploaded_pdf"
                        documents, chunks = split_text_into_documents(pdf_text, source=filename)
                        st.session_state.pdf_documents = documents
                        st.session_state.pdf_chunks = chunks

                    st.success(f"✅ PDF processed! Extracted {len(pdf_text):,} characters into {len(chunks)} chunks with rich metadata.")

                    # Initialize Chroma vector database if API key is provided
                    if api_key and CHROMA_AVAILABLE:
                        with st.spinner("Initializing vector database with Document metadata..."):
                            try:
                                chroma_client, collection, embeddings = initialize_chroma_vectorstore(api_key)
                                if chroma_client and collection and embeddings:
                                    # Add Document objects to vectorstore with metadata
                                    success = add_documents_to_vectorstore(documents, embeddings, collection)
                                    if success:
                                        st.success("🔍 Vector database initialized with embeddings and metadata!")
                                    else:
                                        st.warning("⚠️ Vector database initialized but failed to add documents.")
                                else:
                                    st.warning("⚠️ Failed to initialize vector database. Using fallback search.")
                            except Exception as e:
                                error_msg = f"Error setting up vector database: {str(e)}"
                                st.session_state.last_error = error_msg
                                st.error(error_msg)

                    # Create ReAct agent if API key is provided
                    if api_key:
                        with st.spinner("Setting up your LangGraph ReAct tutor agent..."):
                            try:
                                st.session_state.tutor_agent = create_react_tutor(api_key, pdf_text, chunks)
                                st.success("🤖 LangGraph ReAct Tutor Agent is ready!")
                            except Exception as e:
                                error_msg = f"Error setting up tutor: {str(e)}"
                                st.session_state.last_error = error_msg
                                st.error(error_msg)
                    else:
                        st.warning("⚠️ Please provide OpenAI API key to enable the tutor agent.")
    
    # Show PDF content preview
    if st.session_state.pdf_content:
        st.header("📖 Document Preview")
        with st.expander("View extracted content"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Characters", f"{len(st.session_state.pdf_content):,}")
            with col_b:
                st.metric("Text Chunks", len(st.session_state.pdf_chunks))
            
            preview_text = st.session_state.pdf_content[:1000]
            if len(st.session_state.pdf_content) > 1000:
                preview_text += "..."
            st.text_area("Document content:", preview_text, height=200, disabled=True)
    
    # Debug mode toggle
    st.header("🐛 Debug Settings")
    debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.get("debug_mode", False))
    st.session_state.debug_mode = debug_mode

    # Error status display
    if st.session_state.last_error:
        st.header("⚠️ System Status")
        with st.expander("Last Error Details", expanded=debug_mode):
            st.error("Last Error:")
            st.text(str(st.session_state.last_error))
            if st.button("Clear Error"):
                st.session_state.last_error = None
                st.rerun()
    else:
        if debug_mode:
            st.header("✅ System Status")
            st.success("No recent errors")

    # System info in debug mode
    if debug_mode:
        st.header("🔧 System Info")
        info_data = {
            "Selected Model": st.session_state.selected_model,
            "Response Length": st.session_state.response_length,
            "ChromaDB Available": CHROMA_AVAILABLE,
            "Matplotlib Available": MATPLOTLIB_AVAILABLE,
            "Plotly Available": PLOTLY_AVAILABLE,
            "SymPy Available": SYMPY_AVAILABLE,
            "DuckDuckGo Available": DUCKDUCKGO_AVAILABLE,
            "Vector DB Initialized": st.session_state.chroma_client is not None,
            "Collection Available": st.session_state.collection is not None,
            "Embeddings Available": st.session_state.embeddings is not None,
            "LangGraph ReAct Agent Available": st.session_state.tutor_agent is not None,
            "PDF Processed": bool(st.session_state.pdf_content),
            "Chunks Count": len(st.session_state.pdf_chunks) if st.session_state.pdf_chunks else 0,
            "Code Outputs": len(st.session_state.code_outputs),
            "Plot Outputs": len(st.session_state.plot_outputs)
        }
        for key, value in info_data.items():
            if isinstance(value, bool):
                st.write(f"**{key}:** {'✅' if value else '❌'}")
            else:
                st.write(f"**{key}:** {value}")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat with Your Tutor")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong style="color: #000000;">You:</strong> 
                <span style="color: #000000;">{message["content"]}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong style="color: #000000;">Tutor:</strong> 
                <span style="color: #000000;">{message["content"]}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask your tutor anything about the uploaded document..."):
        if not st.session_state.pdf_content:
            error_msg = "Please upload and process a PDF first."
            st.session_state.last_error = error_msg
            st.warning(error_msg)
        elif not api_key:
            error_msg = "Please provide your OpenAI API key in the sidebar."
            st.session_state.last_error = error_msg
            st.warning(error_msg)
        elif not st.session_state.tutor_agent:
            error_msg = "Please wait for the LangGraph ReAct tutor agent to be set up. Try refreshing the page if this persists."
            st.session_state.last_error = error_msg
            st.warning(error_msg)
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response from LangGraph ReAct agent
            try:
                with st.spinner("LangGraph ReAct agent is thinking and selecting tools..."):
                    # Clear previous errors
                    st.session_state.last_error = None

                    # Invoke the ReAct agent
                    result = st.session_state.tutor_agent(prompt, st.session_state.messages)

                    # Check if result is valid
                    if not result:
                        raise ValueError("Agent returned empty result")

                    if "result" not in result:
                        raise ValueError("Agent result missing 'result' key")

                    tutor_response = result["result"]

                    if not tutor_response or tutor_response.strip() == "":
                        raise ValueError("Agent returned empty response")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": tutor_response})

                # Debug info for successful responses
                with st.sidebar:
                    if st.session_state.get("debug_mode", False):
                        st.success(f"✅ Response generated using: {result.get('tool_used', 'unknown')}")

                st.rerun()

            except Exception as e:
                error_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "agent_available": st.session_state.tutor_agent is not None,
                    "chunks_available": len(st.session_state.pdf_chunks) if st.session_state.pdf_chunks else 0,
                    "api_key_provided": bool(api_key)
                }

                # Store detailed error for debugging
                st.session_state.last_error = f"Agent Error: {error_details}"

                # User-friendly error message
                if "openai" in str(e).lower() or "api" in str(e).lower():
                    error_msg = "🔑 There seems to be an issue with the OpenAI API connection. Please check your API key and try again."
                elif "graph" in str(e).lower() or "invoke" in str(e).lower():
                    error_msg = "🤖 The AI agent encountered an issue. Please try rephrasing your question or restart by uploading the PDF again."
                elif "embed" in str(e).lower() or "vector" in str(e).lower():
                    error_msg = "🔍 Vector search encountered an issue. The system will fall back to keyword search."
                else:
                    error_msg = f"⚠️ I encountered an error: {str(e)}. Please try a simpler question or reload the PDF."

                # Show error in chat
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

                # Show error in sidebar for debugging
                with st.sidebar:
                    st.error("❌ Last Error Details:")
                    st.code(str(error_details), language="json")

                st.rerun()

with col2:
    st.header("📋 Study Features")
    
    if st.session_state.pdf_content and st.session_state.tutor_agent:
        st.markdown("**Quick Actions:**")
        
        def execute_study_query(prompt_text):
            """Execute a study query automatically and display the response."""
            try:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt_text})

                # Get response from autonomous agent
                with st.spinner("Processing..."):
                    result = st.session_state.tutor_agent(prompt_text, st.session_state.messages)
                    tutor_response = result.get("result", "Sorry, I couldn't process that request.")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": tutor_response})
                st.rerun()

            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.session_state.last_error = error_msg
                st.error(error_msg)

        if st.button("📝 Generate Summary"):
            execute_study_query("Can you provide a summary of the main topics in the uploaded document?")

        if st.button("❓ Create Quiz"):
            execute_study_query("Create a quiz to test my understanding of the key concepts.")

        if st.button("🔍 Key Concepts"):
            execute_study_query("What are the most important concepts I should focus on? Give me an overview of the uploaded document.")

        if st.button("💡 Study Tips"):
            execute_study_query("Give me some study tips and strategies for this material.")

        if st.button("📊 Document Overview"):
            execute_study_query("Can you give me an overview of what the uploaded document covers?")

        # Tool buttons
        st.markdown("**🛠️ Advanced Tools:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💻 Code Example"):
                execute_study_query("Show me a code example related to this topic")
            if st.button("📈 Create Plot"):
                execute_study_query("Create a visualization or plot for the data")

        with col_b:
            if st.button("🧮 Show Formula"):
                execute_study_query("Display mathematical formulas related to this topic in the uploaded document")
            if st.button("🌐 Web Search"):
                execute_study_query("Search internet for additional information about this topic")
    else:
        st.info("Upload a PDF and provide API key to access study features.")

# Tool Output Display Areas
if st.session_state.plot_outputs or st.session_state.code_outputs:
    st.markdown("---")
    st.header("🛠️ Tool Outputs")

    # Display plots
    if st.session_state.plot_outputs:
        st.subheader("📊 Generated Plots")
        for i, plot_data in enumerate(st.session_state.plot_outputs):
            with st.expander(f"Plot {i+1}: {plot_data.get('title', 'Untitled')}", expanded=True):
                if plot_data["type"] == "plotly" and PLOTLY_AVAILABLE:
                    st.plotly_chart(plot_data["figure"], use_container_width=True)
                elif plot_data["type"] == "matplotlib":
                    # Decode base64 image
                    import base64
                    img_data = base64.b64decode(plot_data["data"])
                    st.image(img_data, caption=plot_data.get("title", "Generated Plot"))

        if st.button("🗑️ Clear Plots"):
            st.session_state.plot_outputs = []
            st.rerun()

    # Display code examples
    if st.session_state.code_outputs:
        st.subheader("💻 Code Examples")
        for i, code_data in enumerate(st.session_state.code_outputs):
            title = f"Code Example {i+1} ({code_data['language']})"
            if code_data.get('description'):
                title = f"Code Example {i+1}: {code_data['description']}"

            with st.expander(title, expanded=False):
                st.code(code_data["code"], language=code_data["language"])
                if code_data.get("description"):
                    st.markdown(f"**Description:** {code_data['description']}")

        if st.button("🗑️ Clear Code Examples"):
            st.session_state.code_outputs = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
:blue[**How to use:**]

:green[1. Add your OpenAI API key and configure AI model in the sidebar]  
:green[2. Upload a PDF document (textbook, notes, etc.)]  
:green[3. Click "Process PDF" to extract the content]  
:green[4. Start chatting with your AI tutor!]

:orange[**Features:**]  
- :blue[**🤖 Model Selection**:] :blue[Choose from GPT-4o, GPT-4, or GPT-3.5-turbo models]
- :blue[**📏 Response Length**:] :blue[Control detail level from short to comprehensive]
- :blue[**💻 Code Display**:] :blue[View Python code examples with syntax highlighting]
- :blue[**📊 Data Visualization**:] :blue[Create interactive plots and charts]
- :blue[**🧮 Formula Display**:] :blue[Render mathematical equations with LaTeX] 
- :blue[**🌐 Web Search**:] :blue[Search the internet for additional information]  
- :blue[**🔍 Vector Search**:] :blue[Enhanced document search with OpenAI embeddings]  

:blue[The tutor can help you:]  
:green[- Understand concepts]  
:green[- Create quizzes]  
:green[- Summarize sections]  
:green[- Display code examples]  
:green[- Create visualizations]  
:green[- Show formulas]  
:green[- Search the web]  
:green[- Provide study guidance based on your uploaded material]
"""
)