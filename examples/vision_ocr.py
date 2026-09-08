"""Vision + OCR examples (docs-example payloads)."""

from qubrid import QubridClient

client = QubridClient()

vision = client.vision.analyze(
    model="Qwen/Qwen3-VL-Plus",
    prompt="What is in this image? Describe the main elements.",
    image_urls="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
)
print(vision.content)

ocr = client.ocr.extract(
    image_urls="https://example.com/receipt.jpg",  # replace with a real image URL
    # model="tencent/HunyuanOCR",  # default (live-verified)
)
print(ocr.content)
