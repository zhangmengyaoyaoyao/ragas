from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.llms.base import BaseRagasLLM
from ragas.prompt.pydantic_prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM

# type variables for input and output models 定义类型变量
InputModel = t.TypeVar("InputModel", bound=BaseModel) # 这些类型必须是 BaseModel 的子类
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class ExampleStore(ABC):
    @abstractmethod
    def get_examples(
        self, data: BaseModel, top_k: int = 5
    ) -> t.Sequence[t.Tuple[BaseModel, BaseModel]]:
        pass

    @abstractmethod
    def add_example(self, input: BaseModel, output: BaseModel):
        pass


@dataclass
class InMemoryExampleStore(ExampleStore):
    embeddings: BaseRagasEmbeddings
    _examples_list: t.List[t.Tuple[BaseModel, BaseModel]] = field( # field() 函数用于定义数据类的属性
        default_factory=list, repr=False # 默认是空列表，不显示在 repr 中
    )
    _embeddings_of_examples: t.List[t.List[float]] = field(
        default_factory=list, repr=False
    ) 

    """
    该类的主要功能是将输入和输出的 BaseModel 对象转换为嵌入向量，并存储
    """
    def add_example(self, input: BaseModel, output: BaseModel):
        # get json string for input
        input_json = input.model_dump_json() # 调用 BaseModel 类的 model_dump_json() 方法将 input 转换为 json 字符串
        self._embeddings_of_examples.append(self.embeddings.embed_query(input_json)) # 将 input 转换为嵌入向量并添加到 _embeddings_of_examples 列表中
        self._examples_list.append((input, output))

    """
    get_examples 方法用于获取与输入数据最相似的示例
    1. 将输入数据转换为嵌入向量
    2. 计算输入数据与所有示例的余弦相似度
    top_k:最多返回的示例数量
    threshold: 相似度的阈值
    """
    def get_examples(
        self, data: BaseModel, top_k: int = 5, threshold: float = 0.7
    ) -> t.Sequence[t.Tuple[BaseModel, BaseModel]]:
        data_embedding = self.embeddings.embed_query(data.model_dump_json())
        return [
            self._examples_list[i]
            for i in self.get_nearest_examples(
                data_embedding, self._embeddings_of_examples, top_k, threshold
            )
        ]

    """
    get_nearest_examples 方法用于计算输入数据与所有示例的余弦相似度，并返回相似度最高的示例的索引
    """
    @staticmethod
    def get_nearest_examples(
        query_embedding: t.List[float],
        embeddings: t.List[t.List[float]],
        top_k: int = 3,
        threshold: float = 0.7,
    ) -> t.List[int]:
        # Convert to numpy arrays for efficient computation
        query = np.array(query_embedding)
        embed_matrix = np.array(embeddings)

        # Calculate cosine similarity
        similarities = np.dot(embed_matrix, query) / (
            np.linalg.norm(embed_matrix, axis=1) * np.linalg.norm(query) + 1e-8
        )

        # Get indices of similarities above threshold
        valid_indices = np.where(similarities >= threshold)[0]

        # Sort by similarity and get top-k
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:]]

        return top_indices.tolist()

    def __repr__(self):
        return f"InMemoryExampleStore(n_examples={len(self._examples_list)})"

"""
FewShotPydanticPrompt用于支持少样本学习
"""
@dataclass
class FewShotPydanticPrompt(PydanticPrompt, t.Generic[InputModel, OutputModel]):
    example_store: ExampleStore
    top_k_for_examples: int = 5
    threshold_for_examples: float = 0.7

    def __post_init__(self):
        self.examples: t.Sequence[t.Tuple[InputModel, OutputModel]] = []

    def add_example(self, input: InputModel, output: OutputModel):
        self.example_store.add_example(input, output)
    
    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
        retries_left: int = 3,
    ) -> t.List[OutputModel]:
        # Ensure get_examples returns a sequence of tuples (InputModel, OutputModel)
        self.examples = self.example_store.get_examples(data, self.top_k_for_examples)  # type: ignore
        return await super().generate_multiple(
            llm, data, n, temperature, stop, callbacks, retries_left
        )

    """
    from_pydantic_prompt 方法用于从 PydanticPrompt 对象创建 FewShotPydanticPrompt 对象
    """
    @classmethod
    def from_pydantic_prompt(
        cls, # 类方法的第一个参数是类本身
        pydantic_prompt: PydanticPrompt[InputModel, OutputModel],
        embeddings: BaseRagasEmbeddings,
    ) -> FewShotPydanticPrompt[InputModel, OutputModel]:
        # add examples to the example store
        example_store = InMemoryExampleStore(embeddings=embeddings)
        for example in pydantic_prompt.examples:
            example_store.add_example(example[0], example[1]) #example[0]是输入数据，example[1]是输出数据
        few_shot_prompt = cls(
            example_store=example_store,
        )
        # 复制 PydanticPrompt 对象的属性
        few_shot_prompt.name = pydantic_prompt.name
        few_shot_prompt.language = pydantic_prompt.language
        few_shot_prompt.instruction = pydantic_prompt.instruction
        few_shot_prompt.input_model = pydantic_prompt.input_model
        few_shot_prompt.output_model = pydantic_prompt.output_model
        return few_shot_prompt
