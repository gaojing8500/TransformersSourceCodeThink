
## 对目前PTM相关的算法进行预研
* 18年 BERT算法在NLP领域取得的巨大成功，刷新了多项NLP任务记录，但是BERT开启了NLP 新的算法框架的范式 PTM（pre-trained model）+ fine-tune模式，下游的任务只需要很少的数据与低成本的训练时间 既可以取得该行业任务性能要求，而这种算法框架的基础是17年google提出的self-attention transformer网路，位列目前三大深度学习网路之列（CNN RNN Transformer）。

### PTM算法分类
#### Encode-AE
* 依靠transformer编码搭建的框架，采用双向语言模型进行序列建模，经典算法由BERT/ALBERT Roberta，目前大部分主流MLM模型均采用的是这类编码架构，对于语音理解任务效果较好
#### Decode-AR
* 采用transformer decode编码形式，单向语言模型 对于文本生成较好，比如GPT系列
#### Encoder-Decoder
* 采用全体的transformer结构，综合了AE和AR的缺点，同时对语言理解任务和语言生成任务效果较好，比如UniLM

#### PreFix-LM
* Encoder-Decode编码形式的已经变体方案，比如UniLM

#### PLM
* 思路采用的Prefix_LM架构的升级版本，但是又有区别 比如XLNET

#### ELMO
* 采用的传统的RNN结构（包好LSTM GRU等变体），搭建的双向语言模型结构

#### MultiModel
* 目前数据源主要有图片 文本 语音 视频，一般训练的是双模态形式 比如图片-文字，VQA等，还有技术 语音-文本（ASR） 文本-语音(TSS语音合成)，相对来说 目前主流研究双模态形式，比如图片-文本 文本-图片 语音-文本 文本-语音这类标注的训练数据较多，容易获取,从模型结构来说，一般分为双流模型和单流模型。
##### 双模型
* 典型的双流模型包括LXMERT、ViLBERT
![Image](https://pic4.zhimg.com/80/v2-9c4282471a908e1c4bf2161eb5e0f76c.png)

##### 单模型
* 典型的单流模型包括Unicoder-VL、VisualBERT、VL-VERT、UNITER等。
![Image](https://pic4.zhimg.com/80/v2-59b3666cdb159de3d35e0d4a3b3445e0.png)


**采用预训练模型的多模态方法，比不用预训练的传统方法，在应用效果上是有明显提升的。**

**大规模的标准对齐数据比较缺乏，这会严重制约多模态预训练的发展。所以明显需要数据先行，这是发展技术的前提条件；**


### 优秀的开源项目
#### transformers
* 本次先阅读该开源项目所有代码，主要学习PTM框架和相关的算法训练、推理和原理。本次阅读心得主要以代码注释方式在代码位置处注释标明，每一个部分完成会发布一片关于该章节的心得说明



### PTM的未来在哪里？
* 这里只是发起一个思考，多段训练，通识性-领域性-具体的领域应用场景-具体的领域子任务
![Image](https://pic4.zhimg.com/80/v2-cc1290700c12038d8a4729854d1f7d2b.png)

![Image](https://pic4.zhimg.com/80/v2-60c0464c14f1e63e6308826cc76b4500.png)

### 参考文献
[预训练模型的技术演进：乘风破浪的PTM](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247508265&idx=1&sn=8f5c1710dd2cd274e4daf30e27f78886&chksm=fbd71345cca09a536146f34ebf18fd90c549ccd8418331f1a3ae83b0e573eaf1154c449cc545&mpshare=1&&srcid=09238Rtsr1CF2NY0A86CTlK8&sharer_sharetime=1600824236534&sharer_shareid=7470b5f543d9cda449788eaef8277a5c&scene=1&subscene=10000&clicktime=1600824992&enterid=1600824992&version=3.0.30.2006&platform=win&rd2werd=1#wechat_redirect)

[“Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/pdf/2004.10964.pdf)