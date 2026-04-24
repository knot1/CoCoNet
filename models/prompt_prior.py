import torch
import torch.nn.functional as F

try:
    import open_clip
except ImportError:  # pragma: no cover - handled with runtime guard
    open_clip = None

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_PROMPTS = {
    "roads": [
        "roads in an aerial image",
        "streets from above",
        "urban roads in a satellite image",
    ],
    "buildings": [
        "an aerial image of buildings",
        "rooftops in a satellite image",
        "urban buildings from above",
    ],
    "low veg.": [
        "low vegetation in an aerial image",
        "grassland from above",
        "low plants in a satellite image",
    ],
    "trees": [
        "trees in an aerial image",
        "forest canopy from above",
        "tree crowns in a satellite image",
    ],
    "cars": [
        "cars in an aerial image",
        "vehicles from above",
        "parked cars in a satellite image",
    ],
    "clutter": [
        "clutter in an aerial image",
        "miscellaneous objects in a satellite image",
        "urban clutter from above",
    ],
}


def build_prompts(labels, custom_prompts=None):
    prompts = {}
    for label in labels:
        if custom_prompts and label in custom_prompts:
            prompts[label] = custom_prompts[label]
        elif label in DEFAULT_PROMPTS:
            prompts[label] = DEFAULT_PROMPTS[label]
        else:
            prompts[label] = [f"an aerial image of {label}"]
    return prompts


class ClipPromptPrior:
    def __init__(
        self,
        labels,
        prompts=None,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        image_size=224,
        use_ensemble=True,
        device=None,
    ):
        if open_clip is None:
            raise ImportError("open-clip-torch is required for prompt prior.")

        self.labels = list(labels)
        self.prompts = prompts or build_prompts(self.labels)
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_size = image_size
        self.use_ensemble = use_ensemble
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.text_features = None
        self.prompt_class_indices = None
        self._build_text_features()

    def _build_text_features(self):
        prompt_texts = []
        prompt_class_indices = []
        for class_idx, label in enumerate(self.labels):
            prompt_list = self.prompts.get(label, [f"an aerial image of {label}"])
            if not self.use_ensemble:
                prompt_list = prompt_list[:1]
            prompt_texts.extend(prompt_list)
            prompt_class_indices.extend([class_idx] * len(prompt_list))

        tokens = self.tokenizer(prompt_texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)

        self.text_features = text_features
        self.prompt_class_indices = torch.tensor(prompt_class_indices, device=self.device)

    def _prepare_images(self, images):
        if images.size(1) > 3:
            images = images[:, :3]
        images = images.to(self.device)
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor(CLIP_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    @torch.no_grad()
    def compute_prior(self, images):
        images = self._prepare_images(images)
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)

        sims = image_features @ self.text_features.t()
        class_sims = []
        for class_idx in range(len(self.labels)):
            mask = self.prompt_class_indices == class_idx
            class_sims.append(sims[:, mask].mean(dim=1))
        class_sims = torch.stack(class_sims, dim=1)
        return F.softmax(class_sims, dim=1)

    @staticmethod
    def logits_from_prior(prior, eps=1e-6):
        return prior.clamp_min(eps).log()
