import asyncio
import torch
from typing import List, Sequence, Any, Optional
from langchain.schema import Document
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenNativeReranker(BaseDocumentCompressor):
    """
    A custom reranker based on the Qwen/Qwen3-Reranker-0.6B model.
    It fully implements the 'yes'/'no' judgment logic based on Causal LM defined in the official code.
    """
    top_n: int = 7
    model_name: str = 'Qwen/Qwen3-Reranker-0.6B'
    instruction: Optional[str] = 'Given a retrieval query, retrieve relevant passages that answer the query'

    _model: Any
    _tokenizer: Any
    _prefix_tokens: List[int]
    _suffix_tokens: List[int]
    _token_false_id: int
    _token_true_id: int
    _max_length: int = 8192

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        print(f"initializing Qwen Native Re-ranker model: {self.model_name}...")

        # 1. init model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left', trust_remote_code=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # recommend optimization：if CUDA is available，use bfloat16 and flash_attention_2
        # notice that flash_attention_2 used on Linux
        model_kwargs = {'trust_remote_code': True}
        if device == 'cuda':
            model_kwargs['dtype'] = torch.bfloat16
            # model_kwargs['attn_implementation'] = "flash_attention_2"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs).to(device).eval()


        # 2. precompute Prompt's tokens and 'yes'/'no' token ID
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

        print(f"Using Instruction: '{self.instruction}'")
        print("Qwen Native Re-ranker finish initializing。")

    def _format_instruction(self, query: str, doc: str, instruction: str = None) -> str:
        if instruction is None:
            instruction = 'Given a retrieval query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]):
        inputs = self._tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        for i in range(len(inputs['input_ids'])):
            inputs['input_ids'][i] = self._prefix_tokens + inputs['input_ids'][i] + self._suffix_tokens
        inputs = self._tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self._max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs: dict) -> List[float]:
        batch_scores = self._model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self._token_true_id]
        false_vector = batch_scores[:, self._token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def _get_relevant_documents(self, query: str, documents: Sequence[Document]) -> Sequence[Document]:
        """Main process"""
        # 1. create pairs
        pairs = [self._format_instruction(query, doc.page_content, self.instruction) for doc in documents]
        # 2. process inputs
        inputs = self._process_inputs(pairs)
        # 3. compute logits
        scores = self._compute_logits(inputs)
        # 4. merge scores and documents
        scored_docs = zip(scores, documents)
        # 5. sorted by score desc
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        # 6. return top_n
        return [doc for score, doc in sorted_docs[:self.top_n]]

    def compress_documents(
            self, documents:
            Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None
    ) -> Sequence[Document]:
        return self._get_relevant_documents(query, documents)

    async def acompress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query, documents)

