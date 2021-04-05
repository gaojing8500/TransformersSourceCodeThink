## HuggingFace Transformers开源库源码解析-Pipline
* HuggingFace（抱脸） transfomers是目前NLP预训练模型训练重要的开源库之一，基本实现我们目前大部分熟悉的预训练模型代码和模型，如下图所示PTM(来之[邱锡鹏老师论文](https://arxiv.org/pdf/2003.08271.pdf))。
![Image](https://pic4.zhimg.com/80/v2-d99cecc3d36fd2b98cb7b4f885346d21.png)
#### PTM开源项目（来之邱锡鹏老师论文）
![Image](https://pic4.zhimg.com/80/v2-c9c419b9f7c0ac65519061248e5a1a7f.png)

* 今天我们重点对其中非常有名的开源库[HuggingFace Transformers](https://github.com/huggingface/transformers)库进行源码解读，便于自己学习和更加深度的了解PTM算法原理和应用场景

### Transformers Pipline流程
#### 模型缓存（[huggingface 模型仓库](https://huggingface.co/models)）
* 可以手动下载huggingface Transformes仓库的预训练模型（主要config.json ，xxx.bin vocab.txt三个文件），这里**暂时不区分中英文模型**
![Image](https://pic4.zhimg.com/80/v2-b906530290370899e0fe4acff32ef3e6.png)
![Image](https://pic4.zhimg.com/80/v2-4bf116852aeac8b732d52925ef7fedd6.png)
pipline 默认注册模型类和框架配置，目前Pipline支持feature-extraction、sentiment-analysis、question-answering、table-question-answering、fill-mask、translation等下面都有
```python
# Register all the supported tasks here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": TFAutoModel if is_tf_available() else None,
        "pt": AutoModel if is_torch_available() else None,
        "default": {"model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"}},
    },
    "sentiment-analysis": {
        "impl": TextClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
    },
    "ner": {
        "impl": TokenClassificationPipeline,
        "tf": TFAutoModelForTokenClassification if is_tf_available() else None,
        "pt": AutoModelForTokenClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": TFAutoModelForQuestionAnswering if is_tf_available() else None,
        "pt": AutoModelForQuestionAnswering if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilbert-base-cased-distilled-squad", "tf": "distilbert-base-cased-distilled-squad"},
        },
    },
    "table-question-answering": {
        "impl": TableQuestionAnsweringPipeline,
        "pt": AutoModelForTableQuestionAnswering if is_torch_available() else None,
        "tf": None,
        "default": {
            "model": {
                "pt": "google/tapas-base-finetuned-wtq",
                "tokenizer": "google/tapas-base-finetuned-wtq",
                "tf": "google/tapas-base-finetuned-wtq",
            },
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": TFAutoModelForMaskedLM if is_tf_available() else None,
        "pt": AutoModelForMaskedLM if is_torch_available() else None,
        "default": {"model": {"pt": "distilroberta-base", "tf": "distilroberta-base"}},
    },
    "summarization": {
        "impl": SummarizationPipeline,
        "tf": TFAutoModelForSeq2SeqLM if is_tf_available() else None,
        "pt": AutoModelForSeq2SeqLM if is_torch_available() else None,
        "default": {"model": {"pt": "sshleifer/distilbart-cnn-12-6", "tf": "t5-small"}},
    },
    # This task is a special case as it's parametrized by SRC, TGT languages.
    "translation": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelForSeq2SeqLM if is_tf_available() else None,
        "pt": AutoModelForSeq2SeqLM if is_torch_available() else None,
        "default": {
            ("en", "fr"): {"model": {"pt": "t5-base", "tf": "t5-base"}},
            ("en", "de"): {"model": {"pt": "t5-base", "tf": "t5-base"}},
            ("en", "ro"): {"model": {"pt": "t5-base", "tf": "t5-base"}},
        },
    },
    "text2text-generation": {
        "impl": Text2TextGenerationPipeline,
        "tf": TFAutoModelForSeq2SeqLM if is_tf_available() else None,
        "pt": AutoModelForSeq2SeqLM if is_torch_available() else None,
        "default": {"model": {"pt": "t5-base", "tf": "t5-base"}},
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": TFAutoModelForCausalLM if is_tf_available() else None,
        "pt": AutoModelForCausalLM if is_torch_available() else None,
        "default": {"model": {"pt": "gpt2", "tf": "gpt2"}},
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
            "config": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
            "tokenizer": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
        },
    },
    "conversational": {
        "impl": ConversationalPipeline,
        "tf": TFAutoModelForCausalLM if is_tf_available() else None,
        "pt": AutoModelForCausalLM if is_torch_available() else None,
        "default": {"model": {"pt": "microsoft/DialoGPT-medium", "tf": "microsoft/DialoGPT-medium"}},
    },
}
```
各个任务UML如下图所示，以情感分析为例，主要是文本分类任务（**Pycharm UML生成**）
基本是采用_ScikitCompat为抽象类进行扩展
![Image](https://pic4.zhimg.com/80/v2-294001396838ac8f91a61481e102fdb4.png)

**模型如何加载了？又是如何读入内存中了？以BERT模型为例 在model文件夹下面都这两个文件**
**tokenization_bert_fast.py，tokenization_bert.py**
huggingface 仓库预训练模型下载路径均配置在这里，这个文件中**BertTokenizerFast BertTokenizer**又是在哪里被注册的了？
手动下载模型固然好，但是Pipline框架提供模型缓存方式，主要在[file_utils.py](https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py) 中
```python
def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
```
请求文件路径被sha256+etag加密 通过HTTP请求获取，比如下面所示
![Image](https://pic4.zhimg.com/80/v2-839c22f58c37b49e2009154f7faa09af.png)

### 待续
