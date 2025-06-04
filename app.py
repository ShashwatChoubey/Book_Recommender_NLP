import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr



books = pd.read_csv("books_with_emotions.csv")

books["thumbnail"] = np.where(books["thumbnail"].isna(), "cover_not_found.jpg", books["thumbnail"])


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
Chunk = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = Chunk.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding=embeddings)



def recommendations(query, top_n=100, initial_k=100, max_k=1000, step=100,
                    category=None, tone=None, initial_top_k=50, final_top_k=16):
    seen_isbns = set()
    results = []

    current_k = initial_k

    while len(results) < top_n and current_k <= max_k:
        recs = db_books.similarity_search(query, k=current_k)

        for doc in recs:
            raw = doc.page_content.split(":")[0]
            isbn_clean = raw.strip('"')

            if isbn_clean in seen_isbns:
                continue

            book = books[books["isbn13"].astype(str) == isbn_clean]
            if not book.empty:
                results.append(book)
                seen_isbns.add(isbn_clean)

            if len(results) >= top_n:
                break

        current_k += step

    if not results:
        return pd.DataFrame()

    all_recs = pd.concat(results, ignore_index=True)

    if category and category != "All":
        all_recs = all_recs[all_recs["simple_categories"] == category]

    if tone and tone != "All":
        tone_map = {
            "Happy": "joy",
            "Suspenseful": "surprise",
            "Angry": "anger",
            "Horror": "fear",
            "Sad": "sadness"
        }
        mapped_emotion = tone_map.get(tone)
        if mapped_emotion:
            all_recs = all_recs[all_recs["emotion"] == mapped_emotion]

    return all_recs.head(final_top_k)



def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return [("cover_not_found.jpg", "Please enter a book description to get recommendations.")]

    res = recommendations(query=query, category=category, tone=tone)

    if res.empty:
        return [("cover_not_found.jpg",
                 "No books found matching your criteria.\n\nTry:\n• Different keywords\n• Broader category selection\n• Different emotional tone")]

    output = []
    for _, row in res.iterrows():

        title = row["title"] if pd.notna(row["title"]) else "Unknown Title"
        title = str(title).strip()
        if len(title) > 50:
            title = title[:47] + "..."


        authors_raw = row["authors"] if pd.notna(row["authors"]) else "Unknown Author"
        authors_list = str(authors_raw).split(";")
        authors_clean = [author.strip() for author in authors_list if author.strip()]

        if len(authors_clean) == 0:
            authors_str = "Unknown Author"
        elif len(authors_clean) == 1:
            authors_str = authors_clean[0]
        elif len(authors_clean) == 2:
            authors_str = f"{authors_clean[0]} & {authors_clean[1]}"
        else:
            authors_str = f"{authors_clean[0]}, {authors_clean[1]} & {len(authors_clean) - 2} others"


        if len(authors_str) > 35:
            authors_str = authors_str[:32] + "..."


        description = row["description"] if pd.notna(row["description"]) else "No description available."
        if isinstance(description, str) and description.strip():
            # Clean description
            clean_desc = description.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            clean_desc = ' '.join(clean_desc.split())
            if len(clean_desc) > 150:
                clean_desc = clean_desc[:147] + "..."
        else:
            clean_desc = "No description available."

        # Additional metadata
        category_info = row.get("simple_categories", "Uncategorized")
        if pd.isna(category_info):
            category_info = "Uncategorized"

        rating = row.get("average_rating", None)
        rating_str = ""
        if pd.notna(rating) and rating > 0:
            rating_str = f"⭐ {float(rating):.1f}"


        caption_parts = [
            f"📚 {title}",
            f"✍️ {authors_str}",
            "",
            f"📖 {clean_desc}",
            "",
            f"🏷️ {category_info}"
        ]

        if rating_str:
            caption_parts.append(f"📊 {rating_str}")

        caption = "\n".join(caption_parts)
        output.append((row["thumbnail"], caption))

    return output


categories = ["All"] + sorted([cat for cat in books["simple_categories"].dropna().unique() if pd.notna(cat)])
tones = ["All", "Happy", "Suspenseful", "Angry", "Horror", "Sad"]


custom_css = """
/* Main container styling */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Gallery improvements with scroll bar */
.gallery {
    border-radius: 12px !important;
    overflow-y: scroll !important;
    max-height: 800px !important;
}

/* Force scroll bar to be visible */
.gallery::-webkit-scrollbar {
    width: 12px !important;
}

.gallery::-webkit-scrollbar-track {
    background: #f1f1f1 !important;
    border-radius: 6px !important;
}

.gallery::-webkit-scrollbar-thumb {
    background: #888 !important;
    border-radius: 6px !important;
}

.gallery::-webkit-scrollbar-thumb:hover {
    background: #555 !important;
}

.gallery .thumbnail {
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

.gallery .thumbnail:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
}

/* Button styling */
.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
}

/* Input field styling */
.gr-textbox, .gr-dropdown {
    border-radius: 8px !important;
    border: 2px solid #e5e7eb !important;
    transition: border-color 0.3s ease !important;
}

.gr-textbox:focus, .gr-dropdown:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}

/* Section styling */
.gr-form {
    background: rgba(248, 250, 252, 0.8) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    border: 1px solid #e5e7eb !important;
}

/* Title styling */
h1 {
    color: #1f2937 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
}

/* Label styling */
label {
    font-weight: 600 !important;
    color: #374151 !important;
    margin-bottom: 0.5rem !important;
}
"""


with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=custom_css,
        title="📚 Semantic Book Recommender"
) as dashboard:
    # Main title
    gr.Markdown(
        "# 📚 Semantic Book Recommender\n"
        "### Discover your next great read with AI-powered recommendations\n"
        "---"
    )


    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="🔍 Describe the book you're looking for",
                placeholder="e.g., 'A heartwarming story about friendship', 'Mystery set in Victorian London', 'Epic fantasy adventure'...",
                lines=3,
                info="Be as descriptive as possible for better recommendations"
            )

        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📂 Select Category",
                value="All",
                info="Filter by book category"
            )

            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Choose Emotional Tone",
                value="All",
                info="Match your mood"
            )

            submit_button = gr.Button(
                "🔍 Find My Books",
                variant="primary",
                size="lg"
            )


    gr.Markdown("## 📖 Your Personalized Book Recommendations")


    gr.Markdown(
        "💡 **Tips for better results:**\n"
        "• Use descriptive keywords about plot, themes, or mood\n"
        "• Try different category and tone combinations\n"
        "• Be specific about what you're looking for"
    )


    output_gallery = gr.Gallery(
        label="Recommended Books",
        show_label=True,
        columns=4,
        rows=2,
        height=800,
        object_fit="contain",
        show_download_button=False,
        show_share_button=False,
        container=True
    )


    gr.Markdown(
        "---\n"
        "🎯 **How it works:** Our AI analyzes your description and matches it with books that have similar themes, emotions, and content.\n"
        "📚 **Need help?** Try searching for genres like 'romance', 'sci-fi', emotions like 'hopeful', 'dark', or specific themes like 'coming of age'."
    )


    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_gallery,
        show_progress=True
    )


    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_gallery,
        show_progress=True
    )


if __name__ == "__main__":
    dashboard.launch(
        share=False,
        show_error=True,
        inbrowser=True,
        favicon_path=None
    )