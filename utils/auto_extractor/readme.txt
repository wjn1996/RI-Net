## 自动抽取说明：
> 概要：使用spacy完成对文本的分词、实体识别、序列标注、依存关系分析以及指代消解；

&emsp;&emsp;自动抽取的过程包括三个主要步骤，分别是 **分词处理（Tokenization Processing）** 、**三元组构建（Triple Construction）** 以及**token对齐(Token Alignment)**；

#### 分词处理（Tokenization Processing）

使用spacy工具实现分词。定义包括分词、序列标注、实体识别、句法分析和指代消解等pipeline，并依次得到token级别和实体级别的数据。对于token级别的数据，其保存整个文本所有token级别的包括词性、起始位置（字符级别）、对于的实体类型及标签、依存关系等信息；对于实体级别类似。

对于存在指代类问题的token，spacy加入neuralcoref库定义的指代消解方法，对指代类词（he，her，its等）转换为文中对于的实体，并形成一个列表（每个指代的词对应的实体）；

另外，将名词、时间词、数量词等加入到实体列表中，并进行chunk，规则包括：
- somethin OF something 名词分块，将相邻的两个名词或实体作为一个整体
- verbs with multiple words: 'were exhibited' 动词分块，相邻的两动词则可以合并为一个
- verb + adp; verb + part
- adp + verb; part  + verb
- chunk all between LRB- -RRB- (something between brackets) 存在括号描述的
- 两个实体相邻的


#### 三元组构建（Triple Construction）
对于三元组，提取可能的动词、谓词等，作为候选关系词，并构建成三元组，但其中包含很多指代词；根据指代消解的实体列表，指代类的词进行替换，并筛选；最后过滤掉包含停用词的三元组；

#### token对齐(Token Alignment)
需要知道原文中哪些token是包含在三元组中的，因为我们需要对原文按照token级别进行推理，因此在计算任意两个token之间的相关性时，要知道这两个实体分别对应的是哪个三元组，并以此寻找相应的推理路径；

首先遍历三元组，对于每个三元组的头尾实体，分别从token分词中获取对应的字符级别的起始位置，以此计算其在token级别的起始位置，因此可以头实体的token级别开始位置，和尾实体token级别的终止位置，在这个区间内，关系一定在这个区间范围内，依次对相应的token进行统计。建立一个token级别的map表，key为token的编号，value为其所在的三元组编号。最终得到每个token所对齐的三元组以及其所属的位置（头实体为0，关系为1，尾实体为2）。

示例（实际中标点符号也会算在内）：
> **Obama(0) was(1) born(2) in(3) honululu(4) , US.(5) where(6) is(7) in(8) the(9) west(10) .**
> 三元组：
> **[[obama(0), was(1) born(2) in(3), hobululu(4)], [obama(0), was(1) born(2) in(3), US.(5)], [US.(5), in(8), the(9) west(10)]]**
> Obama(0): [[0,0], [1,0]]
> where(5): [[1,2]] 注意，其对应一个指代是US.

每个token都将对应到三元组具体的位置，以便在后期对每个token进行attention计算时可以得到对应的三元组以及路径上去，更新时，如果token是在关系边上，则会根据三元组自动补全，因为有时候单独计算两个token之间是没有什么太大联系的，例如Obama和born，但补全整个三元组[obama(0), was(1) born(2) in(3), hobululu(4)]后，则便有了相关性，以此增强了推理的能力，同时对于指代类词也可以避免推理错误。





