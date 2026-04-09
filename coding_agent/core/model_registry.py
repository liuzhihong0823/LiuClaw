from __future__ import annotations

import json
from pathlib import Path

from ai.models import list_models
from ai.types import Model


class ModelRegistry:
    """在内置模型目录之上叠加用户自定义模型。"""

    def __init__(self, models_file: Path) -> None:
        """初始化模型注册表并立即加载模型集合。"""

        self.models_file = models_file  # 用户模型定义文件。
        self._models = self._load_models()  # 内置模型与用户模型的合并索引。

    def _load_models(self) -> dict[str, Model]:
        """加载内置模型与用户模型并按 ID 建索引。"""

        models = {model.id: model for model in list_models()}
        if not self.models_file.exists():
            return models
        raw = self.models_file.read_text(encoding="utf-8").strip()
        if not raw:
            return models
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("models.json must contain a list")
        for item in data:
            model = Model(
                id=str(item["id"]),
                provider=str(item["provider"]),
                inputPrice=float(item.get("inputPrice", 0.0)),
                outputPrice=float(item.get("outputPrice", 0.0)),
                contextWindow=int(item["contextWindow"]),
                maxOutputTokens=int(item["maxOutputTokens"]),
                metadata=dict(item.get("metadata", {})),
            )
            models[model.id] = model
        return models

    def get(self, model_id: str) -> Model:
        """根据模型 ID 返回模型定义。"""

        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model '{model_id}'") from exc

    def list(self) -> list[Model]:
        """返回按 ID 排序的全部可用模型。"""

        return sorted(self._models.values(), key=lambda item: item.id)
