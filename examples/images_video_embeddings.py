"""Images, video, embeddings (experimental) examples."""

from qubrid import QubridClient

client = QubridClient()

img = client.images.generate(
    model="p-image",
    prompt="a lighthouse at dusk",
    aspect_ratio="16:9",
    output_format="webp",
)
print(img.data[0].url)

edit = client.images.edit(
    model="p-image-edit",
    prompt="add a red flag",
    image="https://example.com/in.png",  # remote URL → JSON; local path → multipart
)
print(edit.data[0].url)

vid = client.videos.generate(model="p-video", prompt="ocean waves at sunset", duration=5)
print(vid.url)

# EXPERIMENTAL: no Qubrid OpenAPI spec for embeddings; OpenAI-compatible passthrough.
emb = client.embeddings.create(model="BAAI/bge-large-en-v1.5", input="hello world")
print(emb.data[0].embedding[:4])
