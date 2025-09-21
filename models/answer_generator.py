import torch
import logging
from typing import List, Tuple
import os
import gc

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using Gemma model or template fallback"""

    def __init__(self, model_id: str = "google/gemma-3-1b-it", cache_dir: str = "models_cache"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

        # Set Hugging Face token
        os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

        logger.info(f"Initializing Answer Generator with {model_id}")

        # Try to load the model, but don't fail if it doesn't work
        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"Could not load generative model: {e}")
            logger.info("Will use template-based answer generation")

    def _load_model(self):
        """Load the Gemma model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            logger.error("transformers library not available")
            return

        logger.info(f"Loading generative model {self.model_id} on {self.device}")

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            token=os.getenv("HF_TOKEN")
        )

        # Configure model loading based on available memory
        if self.device == "cuda":
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
            logger.info(f"Available GPU memory: {gpu_memory:.1f} GB")

            if gpu_memory < 6:
                # Use 4-bit quantization for low memory
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                        quantization_config=quantization_config,
                        device_map="auto",
                        token=os.getenv("HF_TOKEN")
                    )
                except Exception as e:
                    logger.warning(f"Quantization failed: {e}, trying standard loading")
                    raise
            else:
                # Use float16 for better performance
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=os.getenv("HF_TOKEN")
                )
        else:
            # CPU loading
            logger.warning("CUDA not available, loading on CPU (will be slower)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,
                token=os.getenv("HF_TOKEN")
            )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Generative model loaded successfully on {self.device}")

        # Print model info
        if hasattr(self.model, 'num_parameters'):
            total_params = self.model.num_parameters()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Model parameters: {total_params:,}")

    def _clean_memory(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def generate_answer(self, query: str, documents: List, reasoning_steps: List[str]) -> Tuple[str, float, List[str]]:
        """Generate answer based on retrieved documents"""

        if not documents:
            return "I couldn't find relevant information to answer your query. Please try uploading relevant documents or rephrasing your question.", 0.1, []

        sources_used = [f"{doc.title} ({doc.source})" for doc in documents]

        # Prioritize LLM generation if the model is loaded
        if self.model and self.tokenizer:
            try:
                logger.info("Using generative model for answer synthesis")
                return self._generate_llm_answer(query, documents, sources_used)
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}. Falling back to template.")
                # Fallback to template if LLM generation fails for any reason
                return self._generate_template_answer(query, documents, sources_used)

        # Use template as the default if the model was never loaded
        logger.info("Using template-based answer generation as fallback")
        return self._generate_template_answer(query, documents, sources_used)

    def _generate_llm_answer(self, query: str, documents: List, sources_used: List[str]) -> Tuple[
        str, float, List[str]]:
        """Generates a synthesized answer using the loaded language model."""

        # 1. Create a context from the retrieved documents
        context = ""
        for i, doc in enumerate(documents, 1):
            context += f"Source {i} ({doc.title}):\n{doc.content}\n\n"

        # 2. Build a detailed prompt for the model
        prompt = f"""
        Use the following sources to answer the user's question.
        Provide a direct, concise answer and then briefly explain your reasoning based on the provided text.
        Cite the sources you used in your explanation (e.g., [Source 1]).

        **Sources:**
        {context}

        **User Question:**
        {query}

        **Answer:**
        """

        # 3. Tokenize the prompt and generate the response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)

        # 4. Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. Extract only the answer part, removing the prompt
        try:
            answer_part = generated_text.split("Answer:")[1].strip()
        except IndexError:
            # If the model doesn't follow the format, return its full output
            answer_part = generated_text.strip()

        # Calculate a confidence score based on document relevance
        confidence_score = min(0.95, len(documents) * 0.25 + 0.45)
        self._clean_memory()

        return answer_part, confidence_score, sources_used

    def _generate_template_answer(self, query: str, documents: List, sources_used: List[str]) -> Tuple[
        str, float, List[str]]:
        """Template-based answer generation for fallback."""

        answer_parts = [
            f"Based on the retrieved documents, here's what I found regarding your query: **{query}**",
            ""
        ]

        # Extract and display key points from each document
        for doc in documents:
            content_snippet = doc.content[:600] + "..." if len(doc.content) > 600 else doc.content
            answer_parts.append(f"**From '{doc.title}':**\n{content_snippet}")

        # Add a concluding summary
        if len(documents) > 1:
            answer_parts.extend([
                "",
                "**Summary:**",
                f"The information above from {len(documents)} sources provides relevant context for your question. "
                "A comprehensive answer can be formed by synthesizing these points."
            ])

        final_answer = "\n".join(answer_parts)
        # Confidence is generally lower for template answers as there is no synthesis
        confidence_score = min(0.85, len(documents) * 0.25 + 0.35)

        return final_answer, confidence_score, sources_used