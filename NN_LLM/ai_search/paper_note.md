



# title:Researchy Questions: A Dataset of Multi-Perspective, Decompositional Questions for LLM Web Agents
## 背景: 
1. 现有的qa数据集对于llm不再具备挑战性。nlp领域qa的新需求是非单一事实、多角度的问题，需要包含多个信息需求。文章从搜索log中提取这些具备挑战性的复杂知识性question。  
2. 文章提出researchy question，从搜索中提取的分解类多角度query.这些问题会话费用户更多时间，比如更多的点击、session长度等。对于gpt4，这些问题也很具备挑战性  
3. 文章证明，slow thinking技术，比如分解问题成子query比直接回答问题更有优势  

## 动机:
1. 目前的qa benchmark如问某个国家首都的知识性问题，已经不在对llm构成挑战性。大量传统的single-hop qa比如naturla question、triviaqa、webquestion、searchqa、已经多多少少的被llm解决了,answer存在于一句话或者段落中;multi-hop 比如hotpotqa、hybirdqa会更难一些，需要从多篇文章/段落中桥接信息。但是这些问题仍然是事实性问题，sub-question的提取相对容易,并且与人类问题的分布不一致.现有的long-from 非事实性的qa如 eli5，stack exchange、yahoo answer也各有各自的问题
2. chatbot和ai助手的崛起，支持用户问更深、更细微的问题。llm agent的崛起，可以让用户/llm/tool更加紧密配合。最近的dataset已经开始向着完成更具备挑战性的任务转变。越来越多的aigc内容用来训练llm，再用llm(llm-as-a-judge)来评估存在风险：llm会识别不出来他们不知道的信息。即使rag能启动一定的帮助，但这知识将风险转到了是否能召回正确的信息，以及是否正确使用
3. llm处理复杂任务需要slow thinging,一个简单的策略就是迭代的分解问题，将unknown unkdowns分解成known unknowns.子问题需要清晰的知道缺少的信息，如何找到这个信息，一旦找到如果贡献给最后的answer等。
4. 本文提出researchy questions:来研究llm处理复杂问题的表现.在实际中，处理researchy question往往包含将原始问题分解成sub-question来辅助检索、减少丢失信息的风险。 researchy question代表：真是人类会问的广义信息需求. researchy question 是一个qa数据集，来评估qa系统或者llm agent使用必要的工具来完成answer任务的效果。 question分别是解决resharchy question很重要的一步，但是如果定义/评估sub-question的质量仍然不清晰。数据集会给出用户觉得有用的url,希望好的sub-question至少能找到这些用户点击doc里面的信息
## 核心工作：
发布了researchy question包含用户的queyry，包含
1. query的分解：每个sub-question包含2级的任务规划；
2. 每个question，用户的聚合点击url
3. 有序的sub-query对应sub-question用来提交给搜索引擎进行检索
## 相关工作：
1. 迭代rag: react/sef-ask/ircot等
2. agent qa：autogen/webgpt/wenagent等 ; bing chat/youpro/sciphi
## 核心方法:
### researchy question的构造 
researchy question的构造:搜索query包含事实性、浏览类query,需要进行过滤.
1. 搜索日志挖掘:只保留英文、非成人内容。query频次>=50,减少query噪声。知识获取类意图query的过滤：
   1. 语种：英文；
   2. 非成人意图
   3. 频次>=50
   4. query长度：[3,15]
   5. question 意图
   6. 非导航意图
   7. 非本地、预估、地图意图
   8. 非零售购物意图
   9. 非健康医疗意图
   10. 出发answer卡>=1:answer 卡如阿拉丁，结果包含在一段文本中。
   11. 不能出发大量广告
2. 事实性过滤：区分出事实性 vs 非事实性
   1. 从上一步抽出来200K个query，使用gpt3+icl进行打标。然后训练bert-large分类器。分类器对第一步结果进行过滤.
   2. 人工check样本，选定阈值.获取目标query集合
3. 可分解分类器:并不是所有query都需要分解。训练第二个分类器
   1. 使用gpt+prompt进行打标,训练bert-large
   2. 人工check样本，选定阈值
4. 去重复:
    1. 层次聚类进行去重
5. gpt过滤:使用gpt进行打标，针对一个query从8个不同维度给出打分。
   1. gpt进行打分，移除模糊、不完全query。
   2. 移除不安全query等

### researchy questions特性
1. 评估researchy question的困难程度：
   1. 使用gpt进行进行sub-question分解。发现事实性需要的sub-query最少。yresearch question需要的最多。
2. 与用户搜索行为的一致性

### researchy question的answer评估
两种分解-answer方法：cot和 因子分解
1. 因子分解：多query并行，


## 评估方法
## 基线：
## 数据集：
## 效果:
## 消融实验:
## 结论:





# template
# title:
## 背景: 
## 动机:
## 核心工作：
## 相关工作：
## 核心方法:
## 评估方法
## 基线：
## 数据集：
## 效果:
## 消融实验:
## 结论: