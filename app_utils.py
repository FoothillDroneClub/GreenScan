"""Utility helpers for the Streamlit trash classification MVP."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

YOLO_MODEL_ID = "yolov8n.pt"  # default lightweight checkpoint

EFFICIENTNET_VARIANT = os.getenv("EFFICIENTNET_VARIANT", "EfficientNetB4")
_EFFICIENTNET_DEFAULT_SIZES: Dict[str, int] = {
    "EfficientNetB0": 224,
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    "EfficientNetB3": 300,
    "EfficientNetB4": 380,
    "EfficientNetB5": 456,
    "EfficientNetB6": 528,
    "EfficientNetB7": 600,
}
EFFICIENTNET_IMAGE_SIZE = int(
    os.getenv(
        "EFFICIENTNET_IMAGE_SIZE",
        _EFFICIENTNET_DEFAULT_SIZES.get(EFFICIENTNET_VARIANT, 380),
    )
)

BLIP_MODEL_ID = os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
BLIP_ARCHITECTURE = os.getenv("BLIP_ARCHITECTURE", "blip1")
BLIP_DTYPE = os.getenv("BLIP_DTYPE", "auto")
BLIP_MAX_NEW_TOKENS = int(os.getenv("BLIP_MAX_NEW_TOKENS", "25"))
BLIP_NUM_RETURN_SEQUENCES = int(os.getenv("BLIP_NUM_RETURN_SEQUENCES", "3"))

CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
CLIP_DTYPE = os.getenv("CLIP_DTYPE", "auto")
CLIP_PROMPT_TEMPLATE = os.getenv("CLIP_PROMPT_TEMPLATE", "A photo of {}.")
CLIP_MAX_KEYWORDS = int(os.getenv("CLIP_MAX_KEYWORDS", "2"))

# Reduce TensorFlow verbosity for cleaner Streamlit logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CATEGORY_KEYWORDS = [
    {"keywords": ["banana peel", "apple core", "food scrap", "leftovers", "compost"], "category": "organic_food_scraps", "estimated_kg_co2": 0.2},
    {"keywords": ["paper cup", "cardboard", "paper bag", "newspaper", "magazine"], "category": "paper_cardboard", "estimated_kg_co2": 0.5},
    {"keywords": ["plastic bottle", "water bottle", "pet bottle", "plastic cup"], "category": "single_use_plastic", "estimated_kg_co2": 3.3},
    {"keywords": ["plastic wrapper", "plastic packaging", "plastic container"], "category": "mixed_plastic", "estimated_kg_co2": 2.8},
    {"keywords": ["aluminum can", "soda can", "beer can"], "category": "aluminum_can", "estimated_kg_co2": 1.6},
    {"keywords": ["glass bottle", "wine bottle", "glass jar"], "category": "glass_bottle", "estimated_kg_co2": 1.2},
    {"keywords": ["t shirt", "jeans", "cloth", "fabric", "sock"], "category": "textile", "estimated_kg_co2": 5.0},
    {"keywords": ["phone", "laptop", "charger", "tablet"], "category": "electronics", "estimated_kg_co2": 10.0},
    {"keywords": ["battery", "aa battery", "aaa battery"], "category": "battery", "estimated_kg_co2": 3.8},
    {"keywords": ["compostable cup", "bioplastic", "compostable packaging"], "category": "compostable_packaging", "estimated_kg_co2": 0.6},
]

EMISSION_FACTORS: Dict[str, str] = {
    "organic_food_scraps": "451baec4-3aa4-4462-8b4a-6436ac147346",
    "mixed_plastic": "c9d3c74f-ff9b-4c19-858c-786d1ac6df43",
    "single_use_plastic": "c9d3c74f-ff9b-4c19-858c-786d1ac6df43",
    "paper_cardboard": "bf40a5ee-6833-4d4c-b7e9-e0d581aba3a4",
    "aluminum_can": "33df28d6-1b08-44a8-9b86-6d8af1e92786",
    "glass_bottle": "9b8b42a2-8a9c-496c-a920-05d4953ea055",
    "textile": "f9627b5f-77b4-4605-9e1c-207d1f5e792d",
    "electronics": "d0122ce0-44df-43b4-bac7-ccb316f52c57",
    "battery": "1e4f00df-9074-4d77-b492-1be917986996",
    "compostable_packaging": "bf40a5ee-6833-4d4c-b7e9-e0d581aba3a4",
    "other": "c27e40d5-12f8-4aad-95e7-45449391cc55",
}

DEFAULT_ESTIMATE_KG = 2.5


class ClassificationError(RuntimeError):
    """Raised when an image cannot be classified."""


class MissingDependencyError(RuntimeError):
    """Raised when optional dependencies required for classification are missing."""


@dataclass
class ClassificationResult:
    display_name: str
    category: str
    confidence: float
    estimated_kg: float
    raw_label: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.raw_label.encode()).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "category": self.category,
            "confidence": self.confidence,
            "estimated_kg": self.estimated_kg,
            "raw_label": self.raw_label,
        }


@dataclass
class EmissionEstimate:
    category: str
    kg_co2e: float
    points_awarded: int
    source: str


@dataclass
class DetectionResult:
    label: str
    confidence: float
    category: str
    estimated_kg: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "category": self.category,
            "estimated_kg": self.estimated_kg,
        }


@dataclass
class BlipBundle:
    model: Any
    processor: Any
    device: str
    generation_kwargs: Dict[str, Any]
    framework: str = "blip"
    architecture: str = "blip1"


@dataclass
class ClipBundle:
    model: Any
    preprocess: Any
    tokenizer: Any
    device: str
    framework: str = "clip"


_TF_MODULE: Optional[Any] = None
_KERAS_IMAGE_MODULE: Optional[Any] = None
_MOBILENET_CLASS: Optional[Any] = None
_EFFICIENTNET_CLASS: Optional[Any] = None
_DECODE_PREDICTIONS_FN: Optional[Any] = None
_PREPROCESS_INPUT_FN: Optional[Any] = None
_EFFICIENTNET_PREPROCESS_FN: Optional[Callable[[np.ndarray], np.ndarray]] = None
_YOLO_MODEL: Optional[Any] = None
_BLIP_BUNDLE: Optional[BlipBundle] = None
_CLIP_BUNDLE: Optional[ClipBundle] = None


class BlipLoadingError(MissingDependencyError):
    """Raised when BLIP assets cannot be loaded."""


class BlipInferenceError(RuntimeError):
    """Raised when BLIP inference fails."""


class ClipLoadingError(MissingDependencyError):
    """Raised when CLIP assets cannot be loaded."""


def _select_torch_device(torch_module: Any) -> str:
    """Pick the best available device for BLIP execution."""
    if getattr(torch_module, "cuda", None) and torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module, "backends") and getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_blip() -> BlipBundle:
    """Load BLIP model and processor once and reuse across calls."""
    global _BLIP_BUNDLE
    if _BLIP_BUNDLE is not None:
        return _BLIP_BUNDLE

    transformers_spec = importlib.util.find_spec("transformers")
    if transformers_spec is None:
        raise MissingDependencyError(
            "Transformers is required for BLIP inference. Install it via 'pip install transformers>=4.38'."
        )

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise MissingDependencyError(
            "PyTorch is required for BLIP inference. Install it via 'pip install torch torchvision torchaudio'."
        )

    transformers_module = importlib.import_module("transformers")
    torch_module = importlib.import_module("torch")

    architecture = BLIP_ARCHITECTURE.lower().strip()
    if architecture == "blip2":
        processor_cls = getattr(transformers_module, "Blip2Processor", None)
        model_cls = getattr(transformers_module, "Blip2ForConditionalGeneration", None)
    else:
        processor_cls = getattr(transformers_module, "BlipProcessor", None)
        model_cls = getattr(transformers_module, "BlipForConditionalGeneration", None)

    if processor_cls is None or model_cls is None:
        raise MissingDependencyError(
            "Your transformers install does not include the requested BLIP architecture. Upgrade via 'pip install --upgrade transformers'."
        )

    dtype = None
    if BLIP_DTYPE.lower() != "auto" and hasattr(torch_module, BLIP_DTYPE.lower()):
        dtype = getattr(torch_module, BLIP_DTYPE.lower())

    try:
        processor = processor_cls.from_pretrained(
            BLIP_MODEL_ID,
            torch_dtype=dtype,
        )
        model = model_cls.from_pretrained(
            BLIP_MODEL_ID,
            torch_dtype=dtype,
        )
    except (OSError, ValueError) as error:
        raise BlipLoadingError(
            f"Unable to load BLIP checkpoint '{BLIP_MODEL_ID}'. Ensure the model is cached by running "
            "the app once with internet access or pre-download it with 'huggingface-cli download'."
        ) from error

    device = _select_torch_device(torch_module)
    model = model.to(device)
    model.eval()

    generation_kwargs = {
        "max_new_tokens": BLIP_MAX_NEW_TOKENS,
        "num_beams": max(4, BLIP_NUM_RETURN_SEQUENCES),
        "num_return_sequences": BLIP_NUM_RETURN_SEQUENCES,
    }

    framework = "blip2" if architecture == "blip2" else "blip1"
    _BLIP_BUNDLE = BlipBundle(
        model=model,
        processor=processor,
        device=device,
        generation_kwargs=generation_kwargs,
        framework=framework,
        architecture=framework,
    )
    return _BLIP_BUNDLE


def load_clip() -> ClipBundle:
    """Load a CLIP model and processor for text-image similarity scoring."""
    global _CLIP_BUNDLE
    if _CLIP_BUNDLE is not None:
        return _CLIP_BUNDLE

    open_clip_spec = importlib.util.find_spec("open_clip")
    torch_spec = importlib.util.find_spec("torch")
    if open_clip_spec is None or torch_spec is None:
        raise ClipLoadingError(
            "open-clip-torch and torch are required for CLIP inference. Install them via 'pip install open-clip-torch torch'."
        )

    open_clip_module = importlib.import_module("open_clip")
    torch_module = importlib.import_module("torch")

    device = _select_torch_device(torch_module)
    dtype = None
    dtype_name = CLIP_DTYPE.lower().strip()
    if dtype_name != "auto" and hasattr(torch_module, dtype_name):
        dtype = getattr(torch_module, dtype_name)

    try:
        model, _, preprocess = open_clip_module.create_model_and_transforms(
            CLIP_MODEL_ID,
            pretrained="openai",
            device=device,
            precision=str(dtype) if dtype is not None else "fp32",
        )
    except Exception as error:  # noqa: BLE001
        raise ClipLoadingError(
            f"Unable to load CLIP model '{CLIP_MODEL_ID}'. Ensure the checkpoint identifier is valid and weights are available."
        ) from error

    tokenizer = open_clip_module.get_tokenizer(CLIP_MODEL_ID)

    _CLIP_BUNDLE = ClipBundle(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )
    return _CLIP_BUNDLE


def run_clip_ranker(image_bytes: bytes) -> Optional[ClassificationResult]:
    """Score image against category keywords using CLIP similarity."""
    try:
        bundle = load_clip()
    except MissingDependencyError as error:
        raise MissingDependencyError(
            "CLIP dependencies are missing. Install 'open-clip-torch' and 'torch' to enable CLIP fallback."
        ) from error

    torch_module = importlib.import_module("torch")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = bundle.preprocess(image).unsqueeze(0).to(bundle.device)

    with torch_module.no_grad():
        image_features = bundle.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    prompts: List[str] = []
    prompt_metadata: List[Dict[str, Any]] = []
    for entry in CATEGORY_KEYWORDS:
        selected_keywords = entry["keywords"][:CLIP_MAX_KEYWORDS]
        for keyword in selected_keywords:
            prompts.append(CLIP_PROMPT_TEMPLATE.format(keyword))
            prompt_metadata.append(entry)

    if not prompts:
        return None

    tokenized = bundle.tokenizer(prompts)
    if isinstance(tokenized, dict):
        text_inputs = {key: value.to(bundle.device) for key, value in tokenized.items()}
    else:
        text_inputs = tokenized.to(bundle.device)

    with torch_module.no_grad():
        text_features = bundle.model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarities = (image_features @ text_features.T).squeeze(0)
    if similarities.ndim == 0:
        similarities = similarities.unsqueeze(0)

    best_index = int(torch_module.argmax(similarities).item())
    best_similarity = float(similarities[best_index].item())
    normalized_confidence = float(torch_module.sigmoid(torch_module.tensor(best_similarity)).item())
    best_metadata = prompt_metadata[best_index]
    best_prompt = prompts[best_index]

    display_name = best_prompt.replace("A photo of", "").strip().strip(".")
    if not display_name:
        display_name = best_metadata["category"].replace("_", " ")

    return ClassificationResult(
        display_name=display_name.title(),
        category=best_metadata["category"],
        confidence=min(0.99, max(0.05, normalized_confidence)),
        estimated_kg=float(best_metadata.get("estimated_kg_co2", DEFAULT_ESTIMATE_KG)),
        raw_label=best_prompt,
    )


def _normalize_caption(caption: str) -> str:
    """Tidy BLIP captions for display purposes."""
    cleaned = caption.strip()
    if not cleaned:
        return "Uncertain Item"
    return cleaned[0].upper() + cleaned[1:]


def run_blip_consensus(
    image_bytes: bytes,
    bundle: Optional[BlipBundle] = None,
) -> Optional[ClassificationResult]:
    """Run BLIP to generate candidate captions and map to trash categories."""
    active_bundle = bundle or load_blip()
    torch_module = importlib.import_module("torch")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        inputs = active_bundle.processor(images=image, return_tensors="pt")
    except Exception as error:  # noqa: BLE001
        raise BlipInferenceError(f"BLIP preprocessing failed: {error}") from error

    if isinstance(inputs, dict):
        inputs = {key: value.to(active_bundle.device) for key, value in inputs.items()}
    else:
        inputs = inputs.to(active_bundle.device)

    generation_kwargs = dict(active_bundle.generation_kwargs)
    generation_kwargs.update({"output_scores": True, "return_dict_in_generate": True})

    try:
        with torch_module.inference_mode():
            outputs = active_bundle.model.generate(**inputs, **generation_kwargs)
    except Exception as error:  # noqa: BLE001
        raise BlipInferenceError(f"BLIP generation failed: {error}") from error

    captions: List[str] = active_bundle.processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )
    if not captions:
        return None

    sequence_scores = getattr(outputs, "sequences_scores", None)
    if sequence_scores is not None:
        probabilities = torch_module.softmax(sequence_scores, dim=0).cpu().tolist()
    else:
        probabilities = [1.0 / (index + 1) for index in range(len(captions))]

    candidates: List[ClassificationResult] = []
    for index, caption in enumerate(captions):
        normalized_caption = caption.strip()
        if not normalized_caption:
            continue
        match = _match_keyword(normalized_caption)
        confidence = float(probabilities[index]) if index < len(probabilities) else 1.0 / (index + 2)
        confidence = min(0.99, max(0.01, confidence))
        candidates.append(
            ClassificationResult(
                display_name=_normalize_caption(normalized_caption),
                category=match["category"],
                confidence=confidence,
                estimated_kg=float(match.get("estimated_kg_co2", DEFAULT_ESTIMATE_KG)),
                raw_label=normalized_caption,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda result: result.confidence, reverse=True)
    for candidate in candidates:
        if candidate.category != "other":
            return candidate
    return candidates[0]


def _prepare_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes into a model-ready tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    _ensure_tensorflow_loaded()
    array = _KERAS_IMAGE_MODULE.img_to_array(image)
    array = _PREPROCESS_INPUT_FN(array)
    return np.expand_dims(array, axis=0)


def _prepare_image_efficientnet(image_bytes: bytes) -> np.ndarray:
    """Prepare image at the configured EfficientNet resolution."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((EFFICIENTNET_IMAGE_SIZE, EFFICIENTNET_IMAGE_SIZE))
    _ensure_tensorflow_loaded()
    array = _KERAS_IMAGE_MODULE.img_to_array(image)
    preprocess_fn: Callable[[np.ndarray], np.ndarray] = _EFFICIENTNET_PREPROCESS_FN or _PREPROCESS_INPUT_FN
    array = preprocess_fn(array)
    return np.expand_dims(array, axis=0)


def _match_keyword(description: str) -> Dict[str, float]:
    normalized = description.lower()
    for candidate in CATEGORY_KEYWORDS:
        if any(keyword in normalized for keyword in candidate["keywords"]):
            return candidate
    return {"category": "other", "estimated_kg_co2": DEFAULT_ESTIMATE_KG}


def _ensure_tensorflow_loaded() -> None:
    """Load TensorFlow and relevant Keras utilities lazily."""
    global _TF_MODULE, _KERAS_IMAGE_MODULE, _MOBILENET_CLASS, _EFFICIENTNET_CLASS, _DECODE_PREDICTIONS_FN, _PREPROCESS_INPUT_FN, _EFFICIENTNET_PREPROCESS_FN
    if _TF_MODULE is not None:
        return

    tensorflow_spec = importlib.util.find_spec("tensorflow")
    if tensorflow_spec is None:
        raise MissingDependencyError(
            "TensorFlow is required for on-device classification. Install it via 'pip install tensorflow==2.16.1'."
        )

    _TF_MODULE = importlib.import_module("tensorflow")
    keras_applications = importlib.import_module("tensorflow.keras.applications.mobilenet_v2")
    _MOBILENET_CLASS = getattr(keras_applications, "MobileNetV2")
    _DECODE_PREDICTIONS_FN = getattr(keras_applications, "decode_predictions")
    _PREPROCESS_INPUT_FN = getattr(keras_applications, "preprocess_input")

    efficientnet_module = importlib.import_module("tensorflow.keras.applications.efficientnet")
    _EFFICIENTNET_CLASS = getattr(efficientnet_module, EFFICIENTNET_VARIANT)
    _EFFICIENTNET_PREPROCESS_FN = getattr(efficientnet_module, "preprocess_input")

    keras_preprocessing = importlib.import_module("tensorflow.keras.preprocessing.image")
    _KERAS_IMAGE_MODULE = keras_preprocessing


def load_classifier() -> Any:
    """Load a MobileNetV2 classifier with ImageNet weights."""
    _ensure_tensorflow_loaded()
    return _MOBILENET_CLASS(weights="imagenet")


def load_efficientnet() -> Any:
    """Load EfficientNet with ImageNet weights for higher-accuracy classification."""
    _ensure_tensorflow_loaded()
    if _EFFICIENTNET_CLASS is None:
        raise MissingDependencyError(
            "EfficientNet is required for enhanced accuracy. Install TensorFlow with EfficientNet support."
        )
    return _EFFICIENTNET_CLASS(weights="imagenet")


def classify_image(
    model: Any,
    image_bytes: bytes,
    preprocessor: Optional[Callable[[bytes], np.ndarray]] = None,
    decoder: Optional[Callable[[np.ndarray, int], List[List[Tuple[str, str, float]]]]] = None,
) -> ClassificationResult:
    """Run classification and map the top prediction to our categories."""
    input_tensor = (
        preprocessor(image_bytes)
        if preprocessor is not None
        else _prepare_image(image_bytes)
    )
    predictions = model.predict(input_tensor, verbose=0)
    decode_fn = decoder if decoder is not None else _DECODE_PREDICTIONS_FN
    decoded = decode_fn(predictions, top=5)

    if not decoded or len(decoded[0]) == 0:
        raise ClassificationError("No predictions returned from classifier")

    for _, label, probability in decoded[0]:
        readable_label = label.replace("_", " ")
        match = _match_keyword(readable_label)
        if match["category"] != "other" or probability >= 0.2:
            return ClassificationResult(
                display_name=readable_label.title(),
                category=match["category"],
                confidence=float(probability),
                estimated_kg=float(match.get("estimated_kg_co2", DEFAULT_ESTIMATE_KG)),
                raw_label=label,
            )

    # Fallback to the highest-probability class even if we didn't match keywords.
    top_label = decoded[0][0]
    readable_label = top_label[1].replace("_", " ")
    match = _match_keyword(readable_label)
    return ClassificationResult(
        display_name=readable_label.title(),
        category=match["category"],
        confidence=float(top_label[2]),
        estimated_kg=float(match.get("estimated_kg_co2", DEFAULT_ESTIMATE_KG)),
        raw_label=top_label[1],
    )


def _merge_results(
    primary: ClassificationResult,
    secondary: Optional[ClassificationResult],
    confidence_gap_threshold: float,
) -> ClassificationResult:
    """Fuse classifications, preferring agreement and high-confidence results."""
    if secondary is None:
        return primary

    categories_match = primary.category == secondary.category
    raw_label_match = primary.raw_label == secondary.raw_label

    if categories_match or raw_label_match:
        averaged_confidence = min(0.99, (primary.confidence + secondary.confidence) / 2)
        averaged_estimate = (primary.estimated_kg + secondary.estimated_kg) / 2
        preferred = primary if primary.confidence >= secondary.confidence else secondary
        return ClassificationResult(
            display_name=preferred.display_name,
            category=preferred.category,
            confidence=averaged_confidence,
            estimated_kg=averaged_estimate,
            raw_label=preferred.raw_label,
        )

    confidence_gap = abs(primary.confidence - secondary.confidence)
    preferred = primary if primary.confidence >= secondary.confidence else secondary

    if confidence_gap >= confidence_gap_threshold:
        return ClassificationResult(
            display_name=preferred.display_name,
            category=preferred.category,
            confidence=preferred.confidence,
            estimated_kg=preferred.estimated_kg,
            raw_label=preferred.raw_label,
        )

    return ClassificationResult(
        display_name="Uncertain Item",
        category="other",
        confidence=0.0,
        estimated_kg=DEFAULT_ESTIMATE_KG,
        raw_label="uncertain",
    )


def load_yolo() -> Any:
    """Load a YOLOv8 model lazily for object detection."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    ultralytics_spec = importlib.util.find_spec("ultralytics")
    if ultralytics_spec is None:
        raise MissingDependencyError(
            "Ultralytics YOLO is required for object detection. Install it via 'pip install ultralytics'."
        )

    ultralytics_module = importlib.import_module("ultralytics")
    try:
        _YOLO_MODEL = ultralytics_module.YOLO(YOLO_MODEL_ID)
    except Exception as error:  # noqa: BLE001
        raise MissingDependencyError(
            f"Failed to load YOLO checkpoint '{YOLO_MODEL_ID}': {error}"
        ) from error
    return _YOLO_MODEL


def detect_objects(image_bytes: bytes, confidence_threshold: float = 0.25) -> List[DetectionResult]:
    """Run YOLO detection and map results to our trash categories."""
    model = load_yolo()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(image, verbose=False)

    detections: List[DetectionResult] = []
    if not results:
        return detections

    first_result = results[0]
    names = getattr(first_result, "names", {})
    boxes = getattr(first_result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        score_tensor = getattr(box, "conf", None)
        class_tensor = getattr(box, "cls", None)
        if score_tensor is None or class_tensor is None:
            continue
        score = float(score_tensor.cpu().item())
        if score < confidence_threshold:
            continue
        class_index = int(class_tensor.cpu().item())
        label = names.get(class_index, f"class_{class_index}")
        mapped = _match_keyword(label.replace("_", " "))
        detections.append(
            DetectionResult(
                label=label,
                confidence=score,
                category=mapped["category"],
                estimated_kg=float(mapped.get("estimated_kg_co2", DEFAULT_ESTIMATE_KG)),
            )
        )

    # If YOLO found nothing confidently, let vision fallback know.
    if not detections:
        detections.append(
            DetectionResult(
                label="unknown",
                confidence=0.0,
                category="other",
                estimated_kg=DEFAULT_ESTIMATE_KG,
            )
        )

    return detections


def calculate_points(kg_co2: float) -> int:
    """Map a CO₂ estimate to gamified points."""
    safe_kg = max(0.0, float(kg_co2))
    return 1 + int(round(safe_kg * 10))


def fetch_emission_estimate(category: str, fallback_kg: float) -> EmissionEstimate:
    """Reach out to the Climatiq API when possible, otherwise fall back locally."""
    api_key = os.getenv("CLIMATIQ_API_KEY")
    if not api_key:
        return EmissionEstimate(
            category=category,
            kg_co2e=float(fallback_kg),
            points_awarded=calculate_points(fallback_kg),
            source="heuristic",
        )

    factor_id = EMISSION_FACTORS.get(category, EMISSION_FACTORS["other"])
    payload = {
        "emission_factor": {"id": factor_id},
        "parameters": {"energy": 1, "energy_unit": "kWh"},
    }

    try:
        response = requests.post(
            "https://beta3.api.climatiq.io/estimate",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        kg_co2 = float(data.get("co2e", fallback_kg))
        return EmissionEstimate(
            category=category,
            kg_co2e=kg_co2,
            points_awarded=calculate_points(kg_co2),
            source="climatiq",
        )
    except (requests.RequestException, ValueError) as error:
        # Fall back to heuristic estimates if the API is unavailable.
        return EmissionEstimate(
            category=category,
            kg_co2e=float(fallback_kg),
            points_awarded=calculate_points(fallback_kg),
            source=f"heuristic ({error.__class__.__name__})",
        )


def analyze_image(
    primary_model: Optional[Any],
    secondary_model: Optional[Any],
    image_bytes: bytes,
    consistency_threshold: float = 0.15,
    use_blip: bool = False,
) -> Dict[str, object]:
    """Full pipeline: run primary & optional secondary classifiers, then fetch emissions."""
    primary_result: Optional[ClassificationResult] = None
    secondary_result: Optional[ClassificationResult] = None

    if use_blip:
        primary_result = run_blip_consensus(image_bytes)
        if primary_result is None:
            raise ClassificationError(
                "BLIP checkpoint missing. Download it by running the app once with internet access "
                "or pre-fetch using 'huggingface-cli download'."
            )
        if primary_model is not None:
            secondary_result = classify_image(primary_model, image_bytes)
    else:
        if primary_model is None:
            raise ClassificationError("Primary model is not available for classification.")
        primary_result = classify_image(primary_model, image_bytes)
        if secondary_model is not None:
            secondary_result = classify_image(
                secondary_model,
                image_bytes,
                preprocessor=_prepare_image_efficientnet,
            )
        else:
            secondary_result = run_clip_ranker(image_bytes)

    consensus_result = _merge_results(primary_result, secondary_result, consistency_threshold)
    emissions = fetch_emission_estimate(consensus_result.category, consensus_result.estimated_kg)
    return {
        "display_name": consensus_result.display_name,
        "category": consensus_result.category,
        "confidence": consensus_result.confidence,
        "estimated_kg": consensus_result.estimated_kg,
        "emissions": emissions,
        "primary": primary_result.as_dict(),
        "secondary": secondary_result.as_dict() if secondary_result else None,
    }