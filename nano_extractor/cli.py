import click
from nano_extractor.core import NanoPDFExtractor

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--pages', '-p', help='Comma separated page numbers (e.g. 1,2,5)', default=None)
@click.option('--mode', '-m', type=click.Choice(['text', 'hybrid']), default='hybrid', help='Extraction strategy')
@click.option('--format', '-f', type=click.Choice(['json', 'md', 'both']), default='both', help='Export format')
@click.option('--out', '-o', default='output', help='Output file prefix')
def cli(pdf_path, pages, mode, format, out):
    """Extract text, footnotes, and OpenCV-detected charts into Markdown or JSON."""
    page_list = [int(x.strip()) - 1 for x in pages.split(',')] if pages else None
    
    click.echo(f"Processing {pdf_path} in '{mode}' mode...")
    extractor = NanoPDFExtractor()
    data = extractor.process_pdf(pdf_path, pages=page_list, extract_mode=mode)
    extractor.export(data, output_fmt=format, output_prefix=out)
    click.echo(f"Done! Results exported to {out}.*")

if __name__ == '__main__':
    cli()