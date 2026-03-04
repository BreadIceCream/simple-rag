# Text Splitter

## Split markdown

[Split markdown - 文本分流器集成 - LangChain 文档 --- Split markdown - text splitter integration - Docs by LangChain](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)



## Unstructured元数据

我需要先说明：文档里**没有**按你列出的每个扩展名给出一张“完整字段清单”，而是分为：

- 一组“所有文件类型通用”的元数据字段
- 一组“按文件类型额外增加”的元数据字段（只列出有差异的部分）

下面是基于现有文档，能确定的部分。

------

### 1. 所有文件类型通用的元数据字段（适用于 .pdf / .md / .txt / .docx / .html / .htm / .csv）

当源文件中有对应信息时，这些字段都会出现在 `metadata` 中：[[Open-source 元数据](https://docs.unstructured.io/open-source/concepts/document-elements#metadata); [UI 通用元数据](https://docs.unstructured.io/ui/document-elements#common-metadata-fields)]

- `filename`
- `file_directory`
- `filetype`
- `last_modified`
- `coordinates`
- `parent_id`
- `category_depth`
- `image_base64`（仅图像/表格元素且使用 Hi-Res 分割时）
- `image_mime_type`（仅图像元素）
- `text_as_html`（仅表格元素）
- `languages`
- `emphasized_text_contents`
- `emphasized_text_tags`
- `orig_elements`（仅对 chunk 后的元素）
- `is_continuation`（仅对因 max_characters 被拆分的元素）
- `detection_class_prob`（Hi-Res 推理时）

这些字段对你列出的所有扩展名都适用，只是有些字段在特定元素/策略下才会出现（例如 `image_base64` 只在图像/表格元素上出现）。

------

### 2. 按文件类型额外增加的元数据字段

文档只对“有差异”的文件类型列出了额外字段；如果某个类型没出现在这张表里，就表示它**只有上面那组通用字段**，没有特别的额外字段。[[UI 按文件类型元数据](https://docs.unstructured.io/ui/document-elements#additional-metadata-fields-by-file-type); [Open-source 按文档类型元数据](https://docs.unstructured.io/open-source/concepts/document-elements#additional-metadata-fields-by-document-type)]

#### 2.1 `.pdf`

额外字段（在通用字段基础上增加）：[[Open-source 按文档类型元数据](https://docs.unstructured.io/open-source/concepts/document-elements#additional-metadata-fields-by-document-type); [UI 按文件类型元数据](https://docs.unstructured.io/ui/document-elements#additional-metadata-fields-by-file-type)]

- `page_number`
- `links`
- `link_start_indexes`
- `image_path`（仅开源概念文档中提到）
- `image_mime_type`（图像元素）

#### 2.2 `.docx`（Word 文档）

额外字段：[[UI 按文件类型元数据](https://docs.unstructured.io/ui/document-elements#additional-metadata-fields-by-file-type); [Open-source 按文档类型元数据](https://docs.unstructured.io/open-source/concepts/document-elements#additional-metadata-fields-by-document-type)]

- `page_number`
- `header_footer_type`（`primary` / `even_only` / `first_page`，用于页眉页脚元素）

#### 2.3 `.html` / `.htm`（HTML）

额外字段：[[UI 按文件类型元数据](https://docs.unstructured.io/ui/document-elements#additional-metadata-fields-by-file-type); [Open-source 按文档类型元数据](https://docs.unstructured.io/open-source/concepts/document-elements#additional-metadata-fields-by-document-type)]

- `image_url`
- `image_mime_type`（图像元素）
- `link_urls`
- `link_texts`
- `link_start_indexes`

#### 2.4 `.md`（Markdown）

文档中**没有单独列出 Markdown 的额外元数据字段**，因此可以确定的只有通用字段那一组。[[Open-source 元数据](https://docs.unstructured.io/open-source/concepts/document-elements#metadata)]

#### 2.5 `.txt`（纯文本）

同样，纯文本文件没有单独列出的额外字段，因此只有通用字段。[[Open-source 元数据](https://docs.unstructured.io/open-source/concepts/document-elements#metadata)]

#### 2.6 `.csv`

CSV 也没有单独列出的额外字段，因此只有通用字段。[[Open-source 元数据](https://docs.unstructured.io/open-source/concepts/document-elements#metadata)]

------

### 3. 小结（按你给的扩展名汇总）

- `.pdf`：通用字段 + `page_number`, `links`, `link_start_indexes`, `image_path`（开源文档）, `image_mime_type`（图像元素）。
- `.docx`：通用字段 + `page_number`, `header_footer_type`。
- `.html` / `.htm`：通用字段 + `image_url`, `image_mime_type`, `link_urls`, `link_texts`, `link_start_indexes`。
- `.md`：仅通用字段（文档未列出额外字段）。
- `.txt`：仅通用字段。
- `.csv`：仅通用字段。

如果你关心的是“某个具体元素类型”（比如表格、图片、标题）在这些文件里的元数据，我可以再按元素类型帮你拆一层。



## 切分元数据内容

切分后的chunk的metadata中总共有3个id，file_id是所属文件的id（对应数据库中的id），element_id和parent_id是Unstructured返回的id，他们俩是关联的。其中chunk有个属性为id，标识每个chunk。

> 将id从metadata中移除

代码文件的chunk的metadata中只有file_id。

非代码文件metadata包含3个id，使用Unstructured第一次切分的doc有element_id和parent_id，在使用text_splitter切分前设置统一的file_id.

metadata中的index用于表示当前chunk是原文件的第几个chunk，通过file_id + index也可以定位到这个chunk。

splitters元数据表示当前文档切分时所用的全部text_splitter

