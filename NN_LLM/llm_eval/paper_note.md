
# title:G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment
## 背景: 
- llm生成text的质量很难自动化评估,传统的指标入bleu,rouge 与人类的评估相关性很低。  
- 最近的研究表明可以使用LLM作为nlg的评估
- 本文提出G-eval,使用llm结合cot来评估nlg的质量
- https://github.com/nlpyang/geval
- reference-based评估：需要参考进行评估
- reference-free:不需要参考的评估
## 动机:
## 核心工作：
1. 对于NLG(summary/对话响应)任务，llm based的方法超过之前的baseline
2. 基于llm的评估对指令和prompt敏感，cot通过提供guide可以提升llm评估的效果；
3. 通过token的logit做reweighting，llm可以提供细粒度的float得分
4. llm可能会更偏好llm的输出，如果使用llm 评估作为reward，可能会陷入自我强化。
## 相关工作：
最近的研究提出直接使用LLM作为nlg的评估：使用llm针对候选生成进行打分。这种方法的假设是：llm对高质量、流程的输出胡分配更高的权重
1. gptscore
## 核心方法:
g-eval的prompt包含三个部分：
1. 任务描述，评估准则
2. cot：通过llm生成的一系列在中间指令，用来描述详细的评估step
3. score函数，通过调用llm来计算输出端得分。比如基于token 的logit。不同于gpt采用token的概率作为评分函数，g-eval采用form-filling的方法，即模型直接输出得分。但是直接打分有两个问题:
    1. 大量分数集中在特定值，导致打分方差很小，见啥hole与人类评估的相关性
    2. 离散的分值，体现不出差异感。
    3. 本文提出使用分值token的权重乘以具体分值来作为最终的打分。
    4. 对于无法直接输出logits的，通过采用多次来估计token概率
方法：
1. 输入任务描述和评估原则,通过cot生成评估步骤
2. 合并生成的cot，来评估nlg输出
## 评估方法
sperman系数；  
kendall-tau系数
## 基线：
gptscore:使用text生成text的loggit近似为score  
bertscore:基于bertembed评估两段文本的相似性  
moverscore:基于bertscore，加入sof对齐和聚合方法，提升打分鲁棒性  
bastscore:使用bart评估平均的likelihood  
unieval:使用T5，将任务转换为QA任务。
## 数据集：
任务类型：summary和对话响应生成  
summaryEval:评估summary能力，label包含4个指标上的人类打分  
topical-chat:对话回复评估  
qags:summary任务中的幻觉评估
## 效果:
1. 传统的rouge、bleu效果很差  
2. 使用nn学习人类打分的方法，相比传统方法提升很多。
3. g-eval超过了之前是所有方法,在一些温度上也超过了gptscore
4. g-eval之前。unieval效果与人类最一致
## 消融实验:
1. llm评估更偏爱llm生成的answer
2. 加入cot后与人类的一致性更高
3. 概率reweighting的作用：效果不稳定，但是float类型的更友好
   
## 结论:




# title: GPTScore: Evaluate as You Desire
## 背景: 
- 生成任务的评估研究的太少。  
- 本问题提出gptscore: 采用llm的涌现能力，来评估生成的文本。  
- https://github.com/jinlanfu/GPTScore
## 动机:
本文结合llm的zero-shot指令，icl能力来初级复杂的评估任务。  
条件生成概率可以用来评估高质量文本的得分。  
icl/cot/zero-shot指令跟随能力可以通过提供标注样本的方式，来自定义需求。  
gptsocre的动机：生成预训练模型会给符合要求的token分配更高的概率。
## 相关工作：
目前的评估研究，要么关注单一方面；要么虽然关注多维度，但是对维度之间的关系关注太少；其他的方法则以来复杂的训过程、人工标注样本。
## 核心方法:
prompt来源：
1. 对于gpt系列， 使用openai官方提供的
2. 对于其他的指令对齐模型，擦用naturalInstruction中的
3. 得分计算：输出token的概率均值
   - 需要提前准备好输入输出，根据 将目标token的输出logit换算成的loss 
## 评估方法
评估方法：
- sperman系数：评估两个变量的单调关系
- perason系数：评估两个变量的线性关系

评估策略
- sample-level：
- dataset-level
## 基线：
score 方法
1. rouge
2. prism
3. bertscore
4. moverscore
5. dynaeval
6. bartscore
## 数据集：
1. 对话响应；基于对话历史，生成响应
2. 文本摘要：基于提供的文本，生成流畅信息量的总结
3. data2文本：
4. 机器翻译：
## 效果:
## 消融实验:
基于模型的消融实验
1. 无指令ft，无示例
2. 有指令ft，无示例
3. 有指令ft，有示例
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