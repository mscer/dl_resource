# dl_resource


论文note

更倾向于从应用角度：novel 结构；一起巧妙的解决应用问题的方法


多任务：
	mmoe/ple

排序：
	deepFM
召回：
	MINd


CRS:
	A Two-Stage Approach toward Interactive Recommendation
		场景：交互式推荐
		

RS：
	
	Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits  EE+解释性
		场景：将ee机制与解释性联合建模。 不同人对解释性的反应不同，同一个人不同上下文需求的可解释性推荐也不同。
		应用：推荐系统的排序问题？
		高德的特殊性： 地理位置导致的bias，召回受限于地理位置

	Graph Neural Networks in Recommender Systems: A Survey
		场景：rs的任务的核心是通过交互信息和side info 学习有效的user/item表示。 而GNN天然契合这个场景


NLP:
	Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models 
	
	Generalization through Memorization: Nearest Neighbor Language Models
		场景：提升lm的泛化能力
		动机：强假设：表示学习问题要比预测问题更简单
		方法：使用knn召回机制，增强plm。使用召回的top-n训练语料（<sentence，target_word>），来辅助plm的输出;预测时，从top-n对应的target_word中建立下一个此的概率分布Pknn， Pknn和原始plm的P概率做线性插值，得到最终的下一个词的概率分布。

Summary
	Text Summarization with Pretrained Encoders
		场景：summary。 提出将bert应用于提取式和生成式summary的方法
		动机：验证plm在summary上的效果。	相比于word/sentence，summary需要捕捉更长的语义片段（doc-level）。并且对于生成式summary，还需要plm具备生成能力
		方法：doc-level bert encoder。提取式：encoder-only结构； 生成式：encoder-decoder 结构； encoder和decoder建立分离的optimizer； 提出两阶段的训练法昂是 先提取式train，然后进行生成式train。 doc-level:将多个sentence的cls拼接作为新的sequence，来表示doc。bertsum  bertsumext  bertsum+decoder
		贡献：1 证明了doc-level的重要性；2 提出了plm应用于summary的方法；3 这种方法可以作为plugin，进一步提升效果

	
	
		



CIR
	Leading Conversational Search by Suggesting Useful Questions 
		场景：递进式引导；people also ask。假设已满足相关性
		核心指标：useless, 能帮助用户的引导才有用。
		数据:自定义构建，挖掘流程值得借鉴
		方法：bert-based ranker  vs  gpt-based generator（最大似然，不需要负样本）
	
	Generating Clarifying Questions for Information Retrieval 
		场景：澄清式引导；
		数据： 人工规则构造
		方法： lstm encoder_decoder； 或者强化学习

	Asking Clarifying Questions in Open-Domain Information-Seeking Conversations 
		场景：澄清式引导。 判断对结果的置信度，不置信时 进行澄清式的引导
		数据
		方法


Prompt系列
	Prefix-Tuning: Optimizing Continuous Prompts for Generation 
		场景：nlg; 自回归、encoder-decoder
		动机：离散token次优，直接训练科学系的虚拟token，即连续向量
		方法：引入额外的参数作为 魔板， 固定plm知识，只优化prefix向量。每一层都单独训练prefix
		应用：gpt、 bart； 每个新的场景， 只训练prefix即可
	
	GPT Understands, Too 
		场景： gpt应用到nlu
		动机：
		方法：
		应用点：
		
	Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners:
		场景：小plm的few-shot
		动机：离散token构成的模板和标签映射是次优的。 用虚拟token代替离散token，同时让模板和标签映射同时可学习。
		方法： 虚拟token的魔板  虚拟token的label， 以及ft阶段引入mlm；；虚拟token来源于plm自身，不额外引入参数
		应用：分类相关的任务



GNN:	HOW POWERFUL ARE GRAPH NEURAL NETWORKS? （没啥用）
		动机：尽管gnn变革了图表示学习，但是对gnn的表示能力和限制还缺乏足够深入的了解。提出一个理论框架来分析gnn的表示能力。主要分析与wl-test的区别，并提出新结构
		gnn的两个操作：消息传递和更新。 GCN/GAT/GraphSAGE/ 
		

	
	KGAT: Knowledge Graph Attention Network for Recommendation 
		动机：gnn 可以打破样本之间iid的假设； 协同过滤是找相似node； 监督学习（FM/DIN）是寻找相似iterm。GNN可以同时结合这两种方式。 并且gnn可以模拟高阶 多跳交互；  gnn模拟高阶连通性，可以互补协同过滤和传统的fm类的监督学习
		方法：将两种图合二为一，user_item的二部图（传统二部图），以及item-entity的contexti_info图（基于三元组的kg）； 或者还可以扩展 user_人群画像的图？；预测 u和 i之间的相似度
	
	A Graph-Enhanced Click Model for Web Search 
		场景：利用gnn来做点击模型
		方法：gnn来挖掘点击数据，推理出用户的潜在需求模型；
		要解决的问题：传统点击模型的数据稀疏、冷启动
		graph embed: 学习节点的低维表示，同时表示中保存的有节点的上下文和结构信息
		graph的构造和表示：分别构造query-graph和doc-graph.构造两种类型的edge。并且把graph作为插件，融合到点击模型中。graph-embed采用gat学习，多层的gat通过gru得到最终的node表示
		

	pinsage/kgat/ gnn的应用层


点击模型：
	通过挖掘点击数据，来计算query的真实需求，计算query-doc的相关性。现实条件往往受bias等因素影响，导致点击数据不能完全可信。
	PBM:基于位置的模型
	CM:基于瀑布流的模型，点击跳过首次点击

		


IR；  召回和排序两类：
	

	召回的基线：bm25和 平滑的smt
	排序的基线：各种ltr

	term weighing部分:
		基线：tf*idf;textrank; 基于回归的模型
		NN: DeepTR  deepCT  HDCT
	

		
	Global Weighted Self-Attention Network for Web Search
		方法：把bm25的统计term weighting，结合到dssm中。 多域doc
		应用：nn召回； 双塔
	
	Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval 
		应用：召回； 计算term-weighting 跟bm25结合 
		bert+回归layer计算term weighting。但是要拟合的目标需要build；能否直接依赖搜索日志，直接训练，减少label挖掘哪一步？
	
	
	Context-Aware Document Term Weighting for Ad-Hoc Search 
		应用：召回； 计算term-weighting 跟bm25结合 
		方法：deepCT扩展到doc粒度，缓解长度限制；引入label的自动生成策略

	vpcg:Learning Query and Document Relevance from a Web-scale Click Graph 
		应用：排序特征；
		方法：点击二部图，训练query和doc的表示。 bow表示，学到的是term的权重系数

	dssm
		应用：nn排序； 双塔
		数据：点击样本为正样本，未点击样本为负样本，1:4
	cdssm
		应用：nn召回； 双塔

	arc1 arc2(李航的)
		应用：nn排序； interaction-based
	

	de-bert:dual encoder bert
		应用：nn排序； 双塔
	
	drmm
		应用：nn排序
	k-nrm : softmatch
		方法：计算query与doc的term的相似性， 引入kernel来抽取相似性特征。kennel是常见pooling（mean,max）的泛化。   更好的poolling ,有点意思：除了mean,max之外的可导poolling。 不过kernel参数是否也能做成可导的？
		应用：nn 排序；  交叉表示
	Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search 
		方法：
		应用：
	
	Sparse, Dense, and Attentional Representations for Text Retrieval 
		方法：
		应用：

生成模型： gan/vae/diffusion model

Dall-e2	 Hierarchical Text-Conditional Image Generation with CLIP Latents
		方法：将clip与 diffusion models集合。 首先文本-》image embedding；然后image embed-》image
		应用：艺术创作，效果很厉害；
		依赖知识：clip diffusion model/glide Denoising Diffusion Probabilistic Models
	Diffusion Probabilistic Models




	

		
	

		

	
