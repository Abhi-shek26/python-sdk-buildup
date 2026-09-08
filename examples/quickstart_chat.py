"""Qubrid SDK quickstart — chat (non-streaming + streaming)."""

from qubrid import QubridClient

client = QubridClient()  # QUBRID_API_KEY

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Summarize this support ticket into bullet-point next steps."}
    ],
    max_tokens=4096,
    temperature=0.7,
    top_p=1,
)
print(resp.content)

for chunk in client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    stream=True,
):
    # Some SSE events carry no text (e.g. role-only first chunk) — skip them
    # so the output doesn't contain literal "None"s.
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
print()
