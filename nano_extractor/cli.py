"""Command-Line Interface (CLI) for the NanoPDFExtractor module.

Provides a terminal entry point to process PDF documents using a hybrid approach:
deterministic vector text layer extraction coupled with OpenCV bounding box detection
and Gemini Vision models for extracting structured charts, graphs, and visual tables.
"""

import sys
from pathlib import Path
from typing import Optional, List
import click
from nano_extractor.core import NanoPDFExtractor
from nano_extractor.storage import LocalStorageProvider, S3StorageProvider


def parse_page_numbers(pages_str: str) -> List[int]:
    """Parses and validates a comma-separated string of 1-indexed page numbers.

    Converts user-facing 1-indexed page inputs (e.g., "1, 2, 5") into internal 
    0-indexed integers required by `pdfplumber`. Prevents negative indices by 
    clamping minimum values to 0.

    Args:
        pages_str: Comma-separated list of page numbers (e.g., "1,2,5").

    Returns:
        List of 0-indexed page numbers.

    Raises:
        click.BadParameter: If any provided item cannot be parsed as an integer.
    """
    try:
        raw_pages = [int(p.strip()) for p in pages_str.split(",") if p.strip()]
        # Convert 1-indexed user input to 0-indexed internal logic
        parsed_pages = [max(0, p - 1) for p in raw_pages]
        return parsed_pages
    except ValueError as err:
        raise click.BadParameter(
            f"Invalid page numbers format: '{pages_str}'. Expected comma-separated integers (e.g., '1,2,5')."
        ) from err


@click.command()
@click.argument(
    "pdf_path", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--pages", "-p", 
    type=str, 
    default=None, 
    help="Comma-separated 1-indexed page numbers to process (e.g., '1,2,5'). Defaults to all pages."
)
@click.option(
    "--mode", "-m", 
    type=click.Choice(["text", "hybrid"], case_sensitive=False), 
    default="hybrid", 
    show_default=True, 
    help="Extraction strategy: 'text' (vector layers only) or 'hybrid' (vector layers + OpenCV chart/table visual detection)."
)
@click.option(
    "--format", "-f", 
    type=click.Choice(["json", "md", "both"], case_sensitive=False), 
    default="both", 
    show_default=True, 
    help="Output export file format(s)."
)
@click.option(
    "--out", "-o", 
    type=str, 
    default=None, 
    help="Output file prefix/path (without extensions). Defaults to 'test_results/<pdf_name>_extracted'."
)
@click.option(
    "--storage", "-s",
    type=click.Choice(["local", "s3"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Storage provider backend for saved figure crops."
)
@click.option(
    "--local-dir",
    type=str,
    default="output",
    show_default=True,
    help="Directory for local image storage (used when --storage is 'local')."
)
@click.option(
    "--s3-bucket",
    type=str,
    default=None,
    help="S3 bucket name (required when --storage is 's3')."
)
@click.option(
    "--s3-prefix",
    type=str,
    default="crops",
    show_default=True,
    help="Key prefix/folder path inside S3 bucket."
)
@click.option(
    "--s3-region",
    type=str,
    default=None,
    help="AWS region name for S3 bucket (optional)."
)
@click.option(
    "--model",
    type=str,
    default="gemini-2.5-flash",
    show_default=True,
    help="Gemini vision model identifier."
)
def cli(
    pdf_path: Path, 
    pages: Optional[str], 
    mode: str, 
    format: str, 
    out: Optional[str],
    storage: str,
    local_dir: str,
    s3_bucket: Optional[str],
    s3_prefix: str,
    s3_region: Optional[str],
    model: str
) -> None:
    """Extract text, footnotes, tables, and OpenCV-detected visual figures from PDFs.

    PDF_PATH: Path to the target PDF file on disk.
    """
    # Parse page list if provided
    page_list: Optional[List[int]] = parse_page_numbers(pages) if pages else None

    # Determine default output path if unspecified
    if not out:
        out_dir = Path("test_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = str(out_dir / f"{pdf_path.stem}_extracted")
    else:
        output_prefix = out
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Initialize Storage Provider
    if storage.lower() == "s3":
        if not s3_bucket:
            raise click.BadParameter("The '--s3-bucket' option is required when '--storage s3' is selected.")
        
        storage_provider = S3StorageProvider(
            bucket_name=s3_bucket,
            prefix=s3_prefix,
            region_name=s3_region
        )
        storage_info = f"S3 (s3://{s3_bucket}/{s3_prefix})"
    else:
        storage_provider = LocalStorageProvider(base_dir=local_dir)
        storage_info = f"Local ({Path(local_dir).resolve()})"

    click.secho(f"📄 Processing: {pdf_path.name}", fg="cyan", bold=True)
    click.echo(f"  ├─ Mode: {mode.upper()}")
    click.echo(f"  ├─ Format: {format.upper()}")
    click.echo(f"  ├─ Storage: {storage_info}")
    click.echo(f"  ├─ Model: {model}")
    click.echo(f"  └─ Target Pages: {pages if pages else 'All'}")

    try:
        extractor = NanoPDFExtractor(
            model_name=model,
            storage_provider=storage_provider
        )
        
        extracted_data = extractor.process_pdf(
            pdf_path=str(pdf_path), 
            pages=page_list, 
            extract_mode=mode  # type: ignore
        )
        
        extractor.export(
            data=extracted_data, 
            output_fmt=format,  # type: ignore
            output_prefix=output_prefix
        )

        click.secho(f"✨ Successfully exported to: '{output_prefix}.*'", fg="green", bold=True)

    except Exception as err:
        click.secho(f"❌ Error during extraction: {err}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()