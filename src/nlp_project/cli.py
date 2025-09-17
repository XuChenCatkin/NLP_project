from __future__ import annotations
from pathlib import Path
import typer, yaml
from .utils.logging import setup_logging
from .utils.seed import set_seed
from .embeddings.bge import BGEEmbedder
from .retrieval.dense import DenseRetriever, DenseConfig

app = typer.Typer(help="NLP_project command-line interface")

@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option("src/nlp_project/config/default.yaml", help="Path to YAML config"),
):
    cfg = yaml.safe_load(Path(config).read_text()) if Path(config).exists() else {}
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    set_seed(cfg.get("seed", 42))
    ctx.obj = cfg

@app.command()
def embed(text: str, model: str = typer.Option(None, help="HF model name")):
    embedder = BGEEmbedder(model_name=model or "BAAI/bge-base-en-v1.5")
    vec = embedder.encode([text])
    typer.echo(str(vec.shape))

@app.command()
def retrieve(query: str, top_k: int = 5):
    retr = DenseRetriever(DenseConfig(top_k=top_k))
    hits = retr.search(query)
    for h in hits:
        typer.echo(h)

if __name__ == "__main__":
    app()
