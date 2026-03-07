```json
{
      "id": "fa543925-22b3-4778-bc04-278705efcdd0",
      "metadata": {
        "text_length": 11763,
        "path": "D:\\Bread\\College\\AI\\Code\\RAG\\参考资料.md",
        "file_directory": "D:\\Bread\\College\\AI\\Code\\RAG",
        "file_name": "参考资料.md",
        "file_extension": ".md",
        "last_modified": "2026-03-03T12:45:50",
        "is_url": false,
        "document_loader": "TextLoader",
        "file_id": "62c6ea9b-178a-4196-8687-93ed83ca773d",
        "Header 1": "ParentDocumentRetriever",
        "text_splitters": [
          "MarkdownTextSplitterAdapter",
          "ChildTextSplitter"
        ],
        "id": "fa543925-22b3-4778-bc04-278705efcdd0",
        "parent_index": 0,
        "children_ids": [
          "0902f295-8e7e-4c91-ba01-36435e88317d"
        ],
        "children_count": 1
      },
      "page_content": "# ParentDocumentRetriever  \n**`ParentDocumentRetriever` 是一种高级检索器，通过在检索阶段使用“小块”数据确保精度，而在生成阶段返回其对应的“大块”父文档来提供完整的上下文。**  \n它解决了 RAG 中的一个核心矛盾：为了检索精准，分块通常需要很小；但为了让 LLM 生成高质量答案，分块需要足够大以包含完整的逻辑。",
      "type": "Document"
}
```