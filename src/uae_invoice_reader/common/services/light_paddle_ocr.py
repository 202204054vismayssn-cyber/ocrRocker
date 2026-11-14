"""Use lightweight English PaddleOCR models with GPU acceleration."""

import logging
import os
from statistics import median
from threading import Lock
from typing import Any

import numpy as np

from uae_invoice_reader.common.config import Settings
from uae_invoice_reader.common.exceptions import (
    DocumentExtractionError,
    OcrUnavailableError,
)

logger = logging.getLogger("uae_invoice_reader.ocr")


class LightPaddleOcrService:
    """Run mobile text detection and recognition on GPU (with CPU fallback)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: Any | None = None
        self._lock = Lock()

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                cache_directory = self.settings.paddle_model_cache_dir.resolve()
                cache_directory.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_directory))

                target_device = self.settings.paddle_device.lower()
                is_gpu = target_device in ("gpu", "cuda")
                if is_gpu:
                    try:
                        import paddle

                        if not paddle.device.is_compiled_with_cuda():
                            logger.warning(
                                "PADDLE_DEVICE=gpu was requested, but current Paddle installation is not compiled with CUDA. "
                                "Falling back to cpu."
                            )
                            target_device = "cpu"
                            is_gpu = False
                    except (ImportError, OSError) as cuda_err:
                        logger.warning(
                            "Failed to load Paddle CUDA libraries (%s). Falling back to CPU.",
                            cuda_err,
                        )
                        target_device = "cpu"
                        is_gpu = False

                from paddleocr import PaddleOCR

                device_name = "gpu" if is_gpu else "cpu"
                try:
                    # Native modern PaddleOCR (v2.9+, v3.x, PaddleX) on GPU/CPU
                    self._pipeline = PaddleOCR(
                        use_textline_orientation=False,
                        lang=self.settings.paddle_lang or "en",
                        device=device_name,
                    )
                except Exception as init_err:
                    logger.warning("PaddleOCR modern init with device=%s failed (%s). Trying CPU/generic fallback.", device_name, init_err)
                    try:
                        self._pipeline = PaddleOCR(
                            use_textline_orientation=False,
                            lang=self.settings.paddle_lang or "en",
                            device="cpu",
                        )
                    except Exception:
                        try:
                            self._pipeline = PaddleOCR(lang=self.settings.paddle_lang or "en")
                        except Exception as final_err:
                            raise OcrUnavailableError(f"PaddleOCR could not be initialized: {final_err}") from final_err

            except Exception as exc:
                logger.exception("Failed to initialize PaddleOCR pipeline")
                raise OcrUnavailableError(
                    f"PaddleOCR could not be initialized: {exc}"
                ) from exc

        return self._pipeline

    def extract(self, image: np.ndarray) -> str:
        """Recognize text and rebuild readable rows from OCR coordinates."""

        try:
            pipeline = self._get_pipeline()
            extracted_text = ""

            # Modern PaddleOCR method: .predict()
            if hasattr(pipeline, "predict"):
                try:
                    results = pipeline.predict(image)
                except TypeError:
                    results = pipeline.predict(input=image)
                text_parts = [self._result_text(result) for result in results]
                extracted_text = "\n".join(part for part in text_parts if part).strip()

            # Legacy PaddleOCR method: .ocr()
            elif hasattr(pipeline, "ocr"):
                try:
                    ocr_output = pipeline.ocr(image)
                except TypeError:
                    ocr_output = pipeline.ocr(image, cls=False)
                extracted_text = self._parse_standard_ocr(ocr_output)
            else:
                raise DocumentExtractionError("PaddleOCR instance does not support .predict() or .ocr()")

        except OcrUnavailableError:
            raise
        except Exception as exc:
            logger.exception("PaddleOCR failed to parse the image")
            raise DocumentExtractionError(
                f"PaddleOCR failed to parse the image: {exc}"
            ) from exc

        if not extracted_text:
            raise DocumentExtractionError(
                "PaddleOCR returned no readable text"
            )
        return extracted_text

    @classmethod
    def _parse_standard_ocr(cls, ocr_res: Any) -> str:
        """Extract text and bounding boxes from standard PaddleOCR .ocr() result."""
        if not ocr_res:
            return ""

        if isinstance(ocr_res, list) and len(ocr_res) > 0 and isinstance(ocr_res[0], list):
            items = ocr_res[0]
        else:
            items = ocr_res

        boxes: list[tuple[float, float, float, float, str]] = []
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            polygon, text_info = item[0], item[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                cleaned = str(text_info[0]).strip()
            else:
                cleaned = str(text_info).strip()

            points = np.asarray(polygon, dtype=float)
            if not cleaned or points.size < 4:
                continue
            left = float(points[:, 0].min())
            right = float(points[:, 0].max())
            top = float(points[:, 1].min())
            bottom = float(points[:, 1].max())
            boxes.append((top, left, right, max(1.0, bottom - top), cleaned))

        if not boxes:
            return ""

        return cls._render_boxes_to_rows(boxes)

    @classmethod
    def _result_text(cls, result: Any) -> str:
        result_as_json = getattr(result, "json", result)
        if callable(result_as_json):
            result_as_json = result_as_json()
        if not isinstance(result_as_json, dict):
            return ""

        data = result_as_json.get("res", result_as_json)
        if not isinstance(data, dict):
            return ""

        texts = data.get("rec_text") or data.get("rec_texts") or []
        polygons = data.get("text_det_polys") or data.get("rec_polys") or []
        if not isinstance(texts, list):
            return ""
        if not isinstance(polygons, list) or len(polygons) != len(texts):
            return "\n".join(str(text).strip() for text in texts if str(text).strip())

        boxes: list[tuple[float, float, float, float, str]] = []
        for text, polygon in zip(texts, polygons, strict=False):
            cleaned = str(text).strip()
            points = np.asarray(polygon, dtype=float)
            if not cleaned or points.size < 4:
                continue
            left = float(points[:, 0].min())
            right = float(points[:, 0].max())
            top = float(points[:, 1].min())
            bottom = float(points[:, 1].max())
            boxes.append((top, left, right, max(1.0, bottom - top), cleaned))

        if not boxes:
            return ""

        return cls._render_boxes_to_rows(boxes)

    @staticmethod
    def _render_boxes_to_rows(boxes: list[tuple[float, float, float, float, str]]) -> str:
        """Reconstruct rows and columns with alignment heuristics."""
        typical_height = median(box[3] for box in boxes)
        row_tolerance = max(4.0, typical_height * 0.65)
        rows: list[list[tuple[float, float, float, float, str]]] = []
        for box in sorted(boxes, key=lambda value: (value[0], value[1])):
            if not rows:
                rows.append([box])
            else:
                current_top = sum(item[0] for item in rows[-1]) / len(rows[-1])
                if abs(box[0] - current_top) <= row_tolerance:
                    rows[-1].append(box)
                else:
                    rows.append([box])

        rendered_rows: list[str] = []
        column_gap = max(12.0, typical_height * 2.0)
        document_right = max(box[2] for box in boxes)
        document_middle = document_right * 0.5
        far_right_start = document_right * 0.77
        for row in rows:
            ordered = sorted(row, key=lambda value: value[1])
            row_text = ordered[0][4]
            previous_left = ordered[0][1]
            previous_right = ordered[0][2]
            for item in ordered[1:]:
                gap = item[1] - previous_right
                crosses_middle = (
                    document_right >= 500
                    and previous_left < document_middle <= item[1]
                    and gap > typical_height
                )
                enters_far_right = (
                    document_right >= 500
                    and item[1] >= far_right_start
                    and gap > typical_height * 0.75
                )
                separator = (
                    " | "
                    if gap > column_gap or crosses_middle or enters_far_right
                    else " "
                )
                row_text += separator + item[4]
                previous_left = item[1]
                previous_right = max(previous_right, item[2])
            rendered_rows.append(row_text)

        return "\n".join(rendered_rows)
