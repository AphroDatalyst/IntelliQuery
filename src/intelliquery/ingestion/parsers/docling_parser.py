"""Layout-aware parsing with Docling.

Requires the `ingest` extra. Imports are deferred so the API process, which never
parses documents, does not need PyTorch installed.
"""

import io
import logging
from pathlib import Path
from typing import Any

from intelliquery.ingestion.types import BBox, Element, ElementKind, PageRender, ParsedDocument

logger = logging.getLogger(__name__)

# Figures smaller than this fraction of the page are icons and bullets, not evidence.
MIN_FIGURE_PAGE_FRACTION = 0.01
# Tables Docling labels as a table of contents repeat the headings and add noise.
SKIPPED_TABLE_LABELS = {"document_index"}


def _png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class DoclingParser:
    def __init__(self, *, image_scale: float = 2.0, ocr: bool = False) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.generate_page_images = True
        options.generate_picture_images = True
        options.images_scale = image_scale
        options.do_ocr = ocr
        options.do_table_structure = True
        self._scale = image_scale
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse(self, path: Path) -> ParsedDocument:
        from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

        document = self._converter.convert(path).document
        parsed = ParsedDocument()
        page_images: dict[int, Any] = {}

        for page_number, page in sorted(document.pages.items()):
            if page.image is None:
                continue
            image = page.image.pil_image
            page_images[page_number] = image
            parsed.pages.append(PageRender(page_number, image.width, image.height, _png(image)))

        def locate(item: Any) -> tuple[int | None, BBox | None]:
            if not item.prov:
                return None, None
            prov = item.prov[0]
            page = document.pages.get(prov.page_no)
            if page is None:
                return prov.page_no, None
            box = prov.bbox.to_top_left_origin(page_height=page.size.height)
            scale = self._scale
            return prov.page_no, BBox(box.l * scale, box.t * scale, box.r * scale, box.b * scale)

        def crop(page_number: int | None, bbox: BBox | None) -> bytes | None:
            image = page_images.get(page_number) if page_number is not None else None
            if image is None or bbox is None:
                return None
            return _png(image.crop((int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1))))

        for item, _depth in document.iterate_items():
            page_number, bbox = locate(item)
            label = getattr(getattr(item, "label", None), "value", "")

            if isinstance(item, SectionHeaderItem):
                parsed.elements.append(
                    Element(ElementKind.HEADING, item.text, page_number, bbox, level=item.level)
                )
            elif isinstance(item, TableItem):
                if label in SKIPPED_TABLE_LABELS:
                    continue
                parsed.elements.append(
                    Element(
                        ElementKind.TABLE,
                        item.export_to_markdown(doc=document),
                        page_number,
                        bbox,
                        caption=item.caption_text(document),
                        image=crop(page_number, bbox),
                    )
                )
            elif isinstance(item, PictureItem):
                image = page_images.get(page_number) if page_number is not None else None
                if image is not None and bbox is not None:
                    area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
                    if area < MIN_FIGURE_PAGE_FRACTION * image.width * image.height:
                        continue
                picture = item.get_image(document)
                parsed.elements.append(
                    Element(
                        ElementKind.FIGURE,
                        "",
                        page_number,
                        bbox,
                        caption=item.caption_text(document),
                        image=_png(picture) if picture is not None else crop(page_number, bbox),
                    )
                )
            elif isinstance(item, TextItem):
                if label == "title":
                    parsed.elements.append(
                        Element(ElementKind.HEADING, item.text, page_number, bbox, level=0)
                    )
                elif label == "caption":
                    # Captions are attached to their table or figure above.
                    continue
                else:
                    text = f"- {item.text}" if label == "list_item" else item.text
                    parsed.elements.append(Element(ElementKind.TEXT, text, page_number, bbox))

        logger.info(
            "parsed document",
            extra={"pages": len(parsed.pages), "elements": len(parsed.elements)},
        )
        return parsed
