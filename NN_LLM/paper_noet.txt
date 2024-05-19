
Finetuned Language Models Are Zero-Shot Learners（FLAN）
	背景: 
		1. 指令ft后的lm，可以提升他的zero-shot能力（提升理解和输出能力）；指令后的能力甚至高于few-shot
		2. 指令ft成功的关键：ft数据集的大小，模型的scale，自然语言指令的构造
	动机: 
		1. Gpt3等模型在few-shot上效果很好，但是zero-shot效果比较差。一个可能的远因是，zero-shot时候的指令可能与训练时的promot相差比较大；
		2. 动机：采用指令ft来提升模型对指令的理解能力。使用指令作为监督信号来ft任务，提升lm的指令follow能力
	核心工作： 
		1. 本文探索提升llm zero-shot能力的方法：将nlp任务用指令的方式描述后，在进行ft。提升模型对理解的理解能力，进一步提升zero-shot能力。
		2. 实验证明：增加指令ft中的任务cluster(?)可以提升在新任务上发能力，并且只在足够大的模型上才能体现出来（涌现？）。
	相关工作： 
		ft方式：Bert,T5: pretrain+ft
		prompt方式：GPT: pretrain+prompt
		ift方式：FLAN:pretrain+ift（指令ft）；结合ft和prompt
		
	核心方法: 
		1. 任务和模板：将现有的开源数据集处理成指令的形式；62个数据集，处理成包含12个任务簇的一个大数据集中。每一个数据，人工生成10个不同的模板，为了提升多样性，最多会增加三个模板。
		2. 分类任务：增加option到任务中，option用来防止可选项。
		3. 训练细节：
			底座：lambda-pt
			指令ft:
	基线： 
		Gpt
		Lambda-pt
	数据集： 
	效果: 
	消融实验: 
		在需要理解指令上的效果zero-shot很好，甚至超过few-shot的；但是在不需要指令上的数据机上收益很小。
		需要足够大的模型上效果才会好，小模型效果反而变差，猜测原因似乎ft阶段对小模型的参数影响面更大； 而对大模型总体参数量影响较小，同时教会了模型的指令理解能力。100B以上的效果要好一些
		类似icl，构造few-shot的方式，能来来进一步提升。
	结论:
		所以核心仍然follow the train data。 所以当下游任务已经和pretrain一致（不需要指令提示）时，ift效果提升很小。


评估：
Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling
	背景: 
		a. 为了评估train大小和模型大小的对llm的影响，提出了pythia，包含16个llm从70M到12B参数；包含154个ckpt；
		b. 通过评估来发现新的insight
	动机: 
		a. 虽然transformer起的了一些了成功，但是这些模型对how和why而国内然了解的很少。
		b. 考虑从training和scaling角度来理解。Scaling law只是最底层的理解
	核心工作： 
	相关工作： 
	核心方法: 
		a. 16个模型训练：
			i. 训练了两组，每组各8个模型；两组区别是数据是否去重（采用minihash,阈值0.87）
			ii. 模型结构：基本和gpt一致；除了旋转位置编码/flash attention/
			iii. 优化器：adam,zero，更大的batchsize,1024,长度2048，每个batch200W个token;每个模型采用300Btoken训练（去重后的数据只有200Btoken,需要训练1.5倍）
		b. 评估时的发现：
			i. 预料去重对lm效果并没有很清晰的收益；
			ii. 相比opt尽管用了不用结构，效果并没有明显不同
			iii. 多语种上类似，之前的论文结论可能需要重新评测
	基线： 
	数据集： 
	效果: 
		a. 数据bias对模型的影响
			i. 7B左右以上的大模型受影响比较小
		b. 训练顺序的影响
			i. 训练次序对记忆能力影响很小
		c. 预训练时term次数对下游task的影响
			i. 大模型会有这种线性；
	消融实验: 
	结论:







Training language models to follow instructions with human feedback
	背景: 
		更大的lm并不能使lm更好的满足用户意图，模型与使用者没有对齐。本文通过人介入的ft，来提升对齐能力。Ift+rlhf后，1.3B的gpt效果要好于175的GPT3
	动机:
		 lm的目标是预测下一个token，这与人类想要的有用无害目标不一致。采用ft和rlhf的方式来提升对齐效果；
		对齐会存在对齐税问题，在一些任务上效果会下降；
		rlfh后甚至呈现出指令理解上的泛化，即使训练集没包含的指令，rlfh后也能理解
		总体而言，使用人类偏好的ft 可以提升大量的任务效果，并且提升安全性和可靠性
	核心工作： 
		sft数据集：收集数据，
		reward模型：收集pair偏好数据，训练reward模型：
		ppo：使用ppo优化policy
	相关工作： 
	核心方法: 
		底座模型:gpt3
		Sft：一轮后，sft模型已经在验证机上overfit了，但是继续训练可以提升rm得分和人类偏好，最后训练了16轮；
		Rm：使用了6B的模型训练rm，175的会不稳定，不稳定的lm不适合作为rl的价值函数使用。训练阶段，每个prompt的pair偏好放在一个batch内（据说因为如果随机分布在数据集中，容易产生多次梯度，带来overfitt，不太理解），
		ppo
	基线： 
		sft模型和底座模型
	数据集： 
	效果: 
	消融实验: 
	结论:





agent:

ReAct: Synergizing Reasoning and Acting in Language Models；  基础还是llm+icl+cot； cot-sc;webgpt; star
	背景: llm在推理和行动上的研究作为单独的研究方向；本文提出react 引入reasoning和acting，推理帮助track行动计划；action允许llm与外部tool交互。在评测上，通过简单api即可达到很好的效果。
	动机: 
		1 人类智能包含推理和行动两项。 推理综合当前的环境左侧判断（plan），通过行动得到反馈;
		2 cot的限制：cot的内部推理仍然是黑盒，这会限制他的交互式推理能力和更新知识的能力（cot+rag？）
		3目前还没有同时把推理和action结合在一起，协同作用的研究，如果能把他们结合在一起，则可以形成互补。
	核心工作：
		1 提出react:同时结合推理能力和 行动能力；
		2 在实验上证明react的效果，超过reason-only和act-only
		3 消融实验来分析reason和act的作用；
		4 react的限制和潜力
	相关工作：
		reason-only:cot.只推理，缺乏外部信息的观察；
		act-only:web-gpt,缺乏推理，有与react相同的action，也无法得出最终结果
		react错误的case: think可以经过人类编辑
		采用star（Self-Taught Reasoner）的形式， ft react
		
	核心方法: prompt-icl； sft-icl
		动作空间设计：llm可以采取的actin； search、loopup、answer
		react prompt: 使用3-6个作为icl（实现发现更多的没啥作用）。
		组合内外部知识：cot-sc和react的结合。llm来决定采用哪种组合； cot-sc的推理结构更精确；react的推理比较随意。
		
	基线：
		standard
		cot; cot-sc(多次采样然后投票，效果好于cot)
		webgpt
	数据集：
		知识推理：hotpotqa; fever
		决策任务：alfworld;webshop
		
	效果:
		知识推理任务
			prompt-based; ft-based
			1 540b上：无论是prompt还是sft, react都是act-only要好；
			2 小模型上， react prompt-basd 效果最差； 但是sft的 react效果一致超过其他方法， 甚至小模型的sft react也能超过standard；
			3 react 对比cot:幻觉是cot的一大难题；react的推理能力比cot差；react依赖召回有信息量的内容；
			4 prompt-based方法中：react+cot-sc结合的方法效果最好
			5 sft-based方法中：react 效果比其他几个都要好； 3k个sample就行。 并且小模型react sft比大模型prompt的集中方法都要好
			6 cot或者standard 的sft效果远远不如act-only或者react:因为前者要求模型记住知识， 而后者则需要模型与外部交互；

		决策任务：
			多工具选择；
			react一致超过act-only
			
	消融实验:
	结论:


reason:

cot-ft:Large Language Models Are Reasoning Teachers     多样性处理类似cot-sc可以多借鉴
	使用llm的zero-shot cot能力，来减少小模型cot-sft时候的样本？;  另外生成样本时，提出了多样性推理
	背景: cot只在llm上效果比较好；提出ft-cot, 采用llm生成推理样本， 然后小模型sft，发现sft后小模型推理能力提升，而小模型基于prompt的基线效果很差； llm采用zero-shot 生成（think step by step）推理样本； 生成样本是采用多样式生成，即针对每一个结果，产出多个推理过程； 多样性生成，类似cot-st的处理？？
	动机:
		cot依赖llm才能有好的效果；本文探索如何将大模型的cot迁移到小模型上
	核心工作：
	相关工作：
		迁移学习：pretrain-sft 到 pretrain+prompt;
	核心方法:
		cot-ft:
			1 推理生成： zero-shot shot 生成推理和结果；
			2 过滤上一步输出， 组成符合要求的格式；
			3 sft:
			4 多样性推理：使用随机采样， 而不是贪婪来生成多个推理过程。
				
	基线：
		zero-shot
		zero-shot-cot
		ft
		ft-cot
		teacher
		随机
	数据集：
		数学、常识、符号、其他
	效果:
		与 prompt、zero-shot、 zero-shot-cot、sft相比。 虽然zero-shot-cot在175B上效果很好，但是在6B上效果比ft效果还要差； 
		cot-ft vs ft:大部分任务是cot-ft高于ft,有些任务上不如ft； 甚至在一些任务上超过了175b上的zero-shot-cot
		多样性推理：效果更好；样本效率更高；
		
	消融实验:
		影响cot-ft的因素：
	结论:
	

Cot-sc:Self-Consistency Improves Chain of Thought Reasoning in Language Models —》 利用多path推理过程提升推理鲁棒性？
	背景:  替换之前的贪婪采样，随机生成过个推理路径，投票选择最终答案；cot可以提升llm推理能力
	动机: 负责任务的推理过程可以有很多种，最终都能得到争取的答案。
	核心工作：
	相关工作：
		cot
		sample and rerank
		
	核心方法:
		1 用采样方法生成多个cot路径， 用投票法选择最后答案；（要求答案可以被投票，离散场景？）
		2 lm对他生成的内容并不能做出正确错误的区分，一切只是概率。其他方法会采用rerank来提升回答质量。
	基线：
		lambda
		gpt4
		ul2
		cot-icl-prompt; 贪婪decode
	数据集：
		数学推理、常识推理、符号推理。
	效果:
		均使用few-shot-icl的方式进行，不涉及ft;
		数学推理：效果提升显著。均能达到sota.不过都是llm 至少20b
		常识推理和符号推理：5/6达到sota，均比其他基线高。即使是OOD场景（icl的示例与任务无关），效果也更好；
		在一些cot比标准prompt-based效果差的场景，cot-sc效果也更好；
		与其他方法相比：效果都更好
			sample-and-rank;
			beam search;
			ensemble-based
			并且采用topk 采样的多样cot 比基于beam-search的 多样性cot效果更好；多样性是产生收益的关键因素
	消融实验:
		对采样策略 和模型大小的鲁棒性：
			采样策略：都很鲁邦，采样的温度稀释，top的p，topk的k。
			模型大小：10b以上收益大一些。 10b的涌现现象好一些？
		多错误模板的鲁棒性：
			一致性越高，准确性越高。
		非自然语言的和zero-shot-cot：
		
	结论:
		cot-sc可以提升效果，同时可以通过不确定性预估lm输出的一致性。
		缺陷：rt问题。5-10个作为触发点比较合适；
		问题：cot可能会存在错误的推理路径，需要未来解决


rag:

Retrieval-Augmented Generation for Large Language Models: A Survey
	背景:  llm的幻觉问题，时效性问题。 可以考虑ragd的方式提升生成的质量。 naive rag; advanced rag; moular rag； 本文阐述rag的细节和benchmark
	动机:	
		文章总结了超过100个rag研究，分析r/a/g的核心技术， 分别从应用领域和研究领域进行分析；下游任务，数据集，benchmark和评估工具；
	核心工作：
		1 详细的分析rag的最新方法。包括naive rag; advance rag; modular rag;
		2 r/a/g的核心技术
		3 rag的评估方法
	核心方法:
		1 rag类型：
			1 naivie rag:  index/retrieval/generate;
			2 advanced rag: pre-retrieval/ indexing/ post-retrievial/genereate;  召回部分加入query改写和排序部分后进行generate；
			3 modular rag: 更复杂的rag，包含迭代性的 召回和排序。
		naive rag:retrieval-read模式。 
			1 index:从pdf/html/word/markdown中提取内容，转化成uniform plain text;切分成chunk, chunk生成embed，建立到databas中；
			2 retrieval: query转向量， 进行向量召回。检索topk相似度的文档， 补充到prompt中；
			3 query和召回的cunk 构成prompt 提供给llm，做生成。
			4 naive rag的缺陷：
				1 召回问题：精度和召回问题，召回不相关；
				2 生成问题：召回相关时仍然会有幻觉问题；  召回不相关文档、bias问题；
				3 增强问题： 召回信息冗余，可能会导致重复； 过度依赖召回的信息， 导致输出仅仅是模型召回的chunk；
		advanced rag: 加入query rewrite和 rerank提升召回质量
			1  pre_retrieval: :index加入滑窗法，精细分割、插入meta信息,增强候选chunk的质量；query:query改写，纠错、扩展等；
			2 post_retrieval: rerank,选择最相关的文档。
		modular rag:
			1  其他的新modual和pattern；结合memory fusion等
		rag vs ft:
			1 rag属于开卷考试，适合动态场景；
			2 ft属于闭卷考试，适合静态场景；
			3 rag效果超过无监督ft, 无论是旧知识，还是新知识。
		2 retrieval
			召回粒度和召回源类型 影响召回结果。
				1 召回类型：非结构数据(text)；半结构数据(text+lable:pdf;)；结构化数据（kg）;llm-生成的内容；
				2 召回粒度：token/ phrase/sentence/chunk/doc
			index优化
				1 chunk优化：固定长度；small2big: sentence召回，然后把sent前后的sentence也一起给llm
				2 元信息：pagename,file name,时间等，用于约束信息。
				3结构化索引：层级索引；kg索引；
			query优化：
				1 query扩展：多query； sub query;
				2 query改写：
			embed优化：
				1 稀疏稠密混合embed:bm25; 语义相关性；两者互补， bm25提供初始集合，和语义的训练预料； 语义提供term-weighting的作用；
				2 embed的ft:ft来适应新的预料分布。
			adapter:
		3 generate:
			直接把召回的内容给llm，并不是一种有效的方法。 接下来分别从内容侧和llm侧优化生成效果。
			1 内容侧优化： 与人类类似， llm也存在过多关注开头和即为，忽视中间内容的现象。
				reranking：相关性&多样性；规则或者模型；
				内容选择/压缩：过多的内容，可能会导致llm错过关键信息。提取关键信息，提出冗余信息；选择核心信息等。
			2 llm的ft
				ft:适应数据分布
				rlhf:对齐人类偏好
		4 增强处理
			1 多次召回-生成；每次都是用原始的query+生成的内容来召回。
			2 多次修改query召回内容:相比1 每次使用改写后的query
			3 
		5 task和评估
			下游任务：rag的核心人物：qa; 对话系统，信息提取 code 搜索
			评估目标：指标：bleu（ngram精度）/rouge（ngram召回）
			召回质量：mrr/ndcg
			生成质量：
		6 讨论和未来方向
			1 rag vs long上下文：rag是llm不可替代的角色 rt角度和 信息信噪比角度。
			2 rag的鲁棒性：加入rag召回错误的信息，llm该如何兼容。
			3 rag结合ft
			4 rag的scaling law:
			5 核心工具：langchian langindex
				
			
	基线：
	数据集：
	效果:
	消融实验:
	结论:


decode:
Contrastive Search Is What You Need For Neural Text Generation
	背景:  之前的对比搜索论文说， llm各向异性。 但是实验发现仔仔gpt-2 小或者middle上有个现象。基于此，做了进一步分析，发现对比搜索即使不包含对比训练阶段，效果也显著好于其他的decode方法。
	动机: A Contrastive Framework for Neural Text Generation 认为确定性和随机性decoder效果不好时因为各向异性，提出了对比train和对比搜素；但是对llm来讲， 对比train成本太高； 本文重新思考是否大模型存在各向异性；结果发现大模型不存在，从而可以直接使用对比搜索的方法来decode；
	核心工作：
		1 分析llm的各向异性和各向同性问题；
		2 评估对比搜索在llm的效果；
		3 实验证明 对比搜索超过其他decoder效果
	相关工作：
		贪心搜索
		beam search
		topk
		topp
		对比搜索
	核心方法:
	基线：
	数据集：
	效果:
	消融实验:
		1 各向同性各向异性对对比搜索的影响；
		2 各向同性时，不同decoder的影响；超参数对topktopp 对比搜索的影响都很大； 但是 对比搜索在相关性和多样性上最鲁邦，topk和topp 均容易顾此失彼
	结论:
		？？既然各向同性， 为何其他decoder还有问题？？ 对比的作用是啥？  也就是说即使各向同性，也不能每次去最大的。 平衡好相关性和多样性。
		使用对比搜索的前提是各向同性。



A Contrastive Framework for Neural Text Generation
	背景:  现有的decoder策略 确定性的容易产生重复和不自然问题； 随机性策略导致缺乏语义一致性；本文分析原因 ，并提出了对比搜索在位置语义一致性的同时，提升多样性。
	动机: 分析认为 生成结果变差是因为 token的语义空间分布在较狭窄的空间； gpt2的 输出文本的cosine矩阵都非常高（都非常相似？？）；提出simcTG:1 输出是选择最高的，同时惩罚最小的。
	核心工作：
	相关工作：
		确定生成：贪心或者beam search；
		随机方法：
	核心方法:
		1 对比训练：
			在lm的训练中额外引入对比loss，打开token之间的表示距离。mle+对比loss。 
		2 对比搜索：
			decode阶段，选择（输出概率-惩罚项）最大的token（最大的几个token中）
	基线：
		top-p; 结合温度系数；
		评估指标:1 lm质量，用ppl衡量； 2 生成质量。
	数据集：
	效果:
		指标上看挺厉害。 但是必须同时结合对比训练+对比生成；  单一对比生成一般不如top-p
	消融实验:
		对比训练时p的作用：p太小或者太大都不好， 取0.5最好；
		对比搜索vs topp采样：
	结论:
		提出假设，做了验证。 从假设触发提出对比训练和对比搜索，效果很好。


other:
	Adapterfusion
	背景: 融合多个任务的知识时，会出现灾难性遗忘，并且样本平衡不好做。提出adapterfusion，两端的学习算法。 第一个阶段提取知识，第二个阶段知识融合。
	动机:
		两种多任务学习的方式：1 任务之间串行，可能存在灾难性遗忘；2 多任务同时训练，loss如何平衡，样本数量如何平衡？。 并且容易在low resouce上过拟合，high resouce上欠拟合。
		最近adapter的提出，可以不同时更新全部参数，只更新少量参数，其他参数不变。
		本文提出adapter fusion: 分两阶段， 第一个阶段分别训练apater,第二个阶段进行adapter的融合。
		but:融合避免plm阶段知识的灾难性遗忘？
	核心工作：
		1 提出adapterfusion，学习多个任务目标；
		2 实验论证效果；
	相关工作：
		多任务学习方法：
			1 序列ft:按顺序一次学习多个任务； 灾难性遗忘问题，
			2 mtl:多个任务之间的平衡很难， 并且有序新增任务需要重训。
		adapter: 假设新任务的知识 需要少量参数即可。
			引入adapter,进行少量参数更新。
			论文发现adapter设计成两层，分别放在multiheadatt和fnn上效果最好；也有只放在fnn上的
	核心方法:
		知识提取：不同任务分别进行adapter学习；
		知识融合：固定上一轮的参数，组合多个adapter，额外加入fusion（attention）层，来融合不同adapter的知识。 第一个阶段每个adapter使用自己的数据，第二个阶段使用全部数据。 即数据使用两次
		超参数部分：要注意刚开始的adapter不要对模型有大的影响，比如设置v的参数矩阵为对角向量，q和k的参数矩阵随机初始化。
	基线：
	数据集：
	效果:
	消融实验:
		对数据集的依赖
	结论:
