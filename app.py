import pdfplumber
import chromadb
from transformers import ViltProcessor, ViltModel
import torch
from PIL import Image
import io
import numpy as np

DEFAULT_IMAGE_CONTEXT = "image from document"
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_multimodal_embeddings"


def chunk_text_for_vilt(text: str, processor: ViltProcessor, max_length: int) -> list[str]:
    """Split long document text into chunks that fit ViLT's short text window."""
    tokenizer = processor.tokenizer
    content_token_limit = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_token_limit < 1:
        content_token_limit = max_length

    chunks = []
    current_words = []
    current_token_count = 0

    for word in text.split():
        word_token_count = len(tokenizer.tokenize(word))
        if current_words and current_token_count + word_token_count > content_token_limit:
            chunks.append(" ".join(current_words))
            current_words = []
            current_token_count = 0

        current_words.append(word)
        current_token_count += word_token_count

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def get_vilt_inputs(processor: ViltProcessor, text: str, image: Image.Image, max_length: int):
    return processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def tensor_to_embedding(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().numpy().astype(np.float32).flatten().tolist()


def store_embeddings_in_chromadb(
    pdf_path: str,
    text_embeddings: torch.Tensor | None,
    text_chunk_embeddings: list[torch.Tensor],
    text_chunks: list[str],
    image_embeddings: list[torch.Tensor],
    persist_path: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    if text_embeddings is not None:
        ids.append(f"{pdf_path}:text:document")
        embeddings.append(tensor_to_embedding(text_embeddings))
        documents.append(" ".join(text_chunks))
        metadatas.append({
            "pdf_path": pdf_path,
            "modality": "text",
            "kind": "document",
            "chunk_count": len(text_chunks),
        })

    for chunk_index, (chunk, chunk_embedding) in enumerate(zip(text_chunks, text_chunk_embeddings), start=1):
        ids.append(f"{pdf_path}:text:chunk:{chunk_index}")
        embeddings.append(tensor_to_embedding(chunk_embedding))
        documents.append(chunk)
        metadatas.append({
            "pdf_path": pdf_path,
            "modality": "text",
            "kind": "chunk",
            "chunk_index": chunk_index,
        })

    for image_index, image_embedding in enumerate(image_embeddings, start=1):
        ids.append(f"{pdf_path}:image:{image_index}")
        embeddings.append(tensor_to_embedding(image_embedding))
        documents.append(f"Image {image_index} extracted from {pdf_path}")
        metadatas.append({
            "pdf_path": pdf_path,
            "modality": "image",
            "kind": "image",
            "image_index": image_index,
        })

    if not ids:
        print("No embeddings to store in ChromaDB.")
        return collection

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Stored {len(ids)} embeddings in ChromaDB collection '{collection_name}' at '{persist_path}'.")
    return collection


def get_pdf_multimodal_embeddings_with_vilt(pdf_path: str, model_name: str = "dandelin/vilt-b32-mlm"):
    """
    Extracts text and images from a PDF and generates embeddings for them using a ViLT model.

    Args:
        pdf_path (str): The path to the PDF file.
        model_name (str): The name of the ViLT model to use from Hugging Face.

    Returns:
        tuple: A tuple containing:
            - text_embeddings (torch.Tensor or None): Embeddings for the extracted text.
            - text_chunk_embeddings (list of torch.Tensor): Embeddings for each text chunk.
            - text_chunks (list[str]): Text chunks used to generate chunk embeddings.
            - image_embeddings (list of torch.Tensor): A list of embeddings for each extracted image.
    """
    all_text = []
    extracted_images = []
    skipped_images = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    all_text.append(text)

                # Extract images
                # pdfplumber extracts images as dictionaries with raw data
                for img_data in page.images:
                    stream = img_data.get("stream")
                    if stream is None:
                        skipped_images += 1
                        continue

                    try:
                        img_bytes = stream.get_data()
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        extracted_images.append(pil_image)
                    except Exception:
                        skipped_images += 1
                            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None, [], [], []

    if skipped_images:
        print(f"Skipped {skipped_images} PDF image streams that Pillow could not decode directly.")

    # Initialize ViLT processor and model
    try:
        processor = ViltProcessor.from_pretrained(model_name)
        model = ViltModel.from_pretrained(model_name)
        max_text_length = model.config.max_position_embeddings
    except Exception as e:
        print(f"Error loading ViLT model or processor: {e}")
        return None, [], [], []

    # --- 1. Get Text Embeddings ---
    text_embeddings = None
    text_chunk_embeddings = []
    document_text = " ".join(all_text)
    text_chunks = []
    if all_text:
        text_chunks = chunk_text_for_vilt(document_text, processor, max_text_length)
        # ViLT requires an image, so use a dummy black image for text-only embedding
        dummy_image = Image.new('RGB', (224, 224), color = 'black')

        chunk_embeddings = []
        for chunk in text_chunks:
            inputs = get_vilt_inputs(processor, chunk, dummy_image, max_text_length)
            with torch.no_grad():
                outputs = model(**inputs)
            chunk_embedding = outputs.pooler_output
            chunk_embeddings.append(chunk_embedding)
            text_chunk_embeddings.append(chunk_embedding)

        text_embeddings = torch.cat(chunk_embeddings, dim=0).mean(dim=0, keepdim=True)
        print(f"Processed {len(text_chunks)} text chunks.")
        print(f"Generated text embedding of shape: {text_embeddings.shape}")
    else:
        print("No text found in the PDF.")

    # --- 2. Get Image Embeddings ---
    image_embeddings = []
    if extracted_images:
        print(f"Found {len(extracted_images)} images in the PDF.")
        for i, img in enumerate(extracted_images):
            # For image embeddings, we can pair the image with the overall document text
            # or a specific caption if available (which pdfplumber doesn't directly provide).
            # Using document_text provides context for the image.
            # If no text was extracted, we can use a dummy text or just process the image.
            
            context_text = text_chunks[0] if text_chunks else DEFAULT_IMAGE_CONTEXT
            
            try:
                # Ensure image is in RGB format for ViLT
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                inputs = get_vilt_inputs(processor, context_text, img, max_text_length)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # pooler_output is a good aggregated representation for the image+text pair
                image_embeddings.append(outputs.pooler_output)
                print(f"Generated embedding for image {i+1} of shape: {outputs.pooler_output.shape}")
            except Exception as img_embed_e:
                print(f"Error generating embedding for image {i+1}: {img_embed_e}")
    else:
        print("No images found in the PDF.")

    return text_embeddings, text_chunk_embeddings, text_chunks, image_embeddings

# --- Example Usage ---
if __name__ == "__main__":
    # To test this, you'll need a PDF file named 'your_document_with_images.pdf'
    # in the same directory as this script, or provide a full path.
    # This PDF should contain both text and some images.

    # If you don't have one handy, you can create a simple one with text and an image
    # using a library like ReportLab. For example:
    # from reportlab.pdfgen import canvas
    # from reportlab.lib.pagesizes import letter
    # from reportlab.lib.utils import ImageReader
    #
    # c = canvas.Canvas("your_document_with_images.pdf", pagesize=letter)
    # c.drawString(100, 750, "This is a sample document with text and an image.")
    # c.drawString(100, 730, "The image below illustrates a concept.")
    # # You'll need an actual image file (e.g., 'sample_image.png') for this to work
    # try:
    #     img = ImageReader('sample_image.png') # Make sure 'sample_image.png' exists!
    #     c.drawImage(img, 100, 400, width=200, height=150)
    # except FileNotFoundError:
    #     print("Warning: 'sample_image.png' not found. Cannot add image to dummy PDF.")
    # c.save()

    pdf_file_path = "rag.pdf" # Make sure this PDF file exists and has images!

    print(f"Attempting to generate multimodal embeddings for: {pdf_file_path}")
    text_embeds, text_chunk_embeds, text_chunks, img_embeds_list = get_pdf_multimodal_embeddings_with_vilt(pdf_file_path)

    if text_embeds is not None:
        print("\n--- Text Embeddings ---")
        print("Text Embeddings shape:", text_embeds.shape)
        print("First 5 text embedding values:\n", text_embeds[0, :5])
    else:
        print("\nNo text embeddings generated.")

    if img_embeds_list:
        print(f"\n--- Image Embeddings ({len(img_embeds_list)} images found) ---")
        for i, img_embed in enumerate(img_embeds_list):
            print(f"Image {i+1} Embedding shape:", img_embed.shape)
            print(f"First 5 values for Image {i+1} embedding:\n", img_embed[0, :5])
    else:
        print("\nNo image embeddings generated.")

    store_embeddings_in_chromadb(
        pdf_path=pdf_file_path,
        text_embeddings=text_embeds,
        text_chunk_embeddings=text_chunk_embeds,
        text_chunks=text_chunks,
        image_embeddings=img_embeds_list,
    )
