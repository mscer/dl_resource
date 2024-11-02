



# title: GPTScore: Evaluate as You Desire
## 背景: 
- 生成任务的评估研究的太少。  
- 本问题提出gptscore: 采用llm的涌现能力，来评估生成的文本。  
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