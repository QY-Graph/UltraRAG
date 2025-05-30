short_dependency_prompt={
    "Chinese":{
        "question":"你非常擅长从文本中提取问题。基于提供的内容，构建三个直接基于文本内容的问题，问题应该能够探索文本中的关键主题、角色关系、事件细节或时间顺序。每个问题之间用换行符分隔。请确保问题的表达清楚地指向文本中的特定信息，避免使用模糊或过于宽泛的引用。同时，强调直接引用或文本中的具体细节，以增加问题的准确性和深度。\n\n不要包含这些词汇 {filter_words}。问题的答案应该能够用几句话或更少的内容来回答。\n\n内容:{content}\n\n回答:",
        "question_pro":"你将会收到一段来自某个专业领域的文本。基于这段文本，构建三个聚焦于该领域关键概念、术语或流程的具体问题。每个问题之间用换行符分隔。问题的答案应当仅通过文本中提供的信息来作答。\n\n文本: {content}\n\n问题:",
        "question_example":"你非常擅长从文本中提取问题。基于提供的内容，构建三个直接基于文本内容的问题，问题应该能够探索文本中的关键主题、角色关系、事件细节或时间顺序。每个问题之间用换行符分隔。请确保问题的表达清楚地指向文本中的特定信息，避免使用模糊或过于宽泛的引用。同时，强调直接引用或文本中的具体细节，以增加问题的准确性和深度。\n\n不要包含这些词汇 {filter_words}。问题的答案应该能够用几句话或更少的内容来回答。\n\n尽量用名字代替代词\n\n如果是关系词，如孩子或父母，必须注明它属于谁\n\n以下是参考的问题风格，但生成的问题内容不能相同。\n\n示例问题风格: {example_question}\n\n内容: {content}\n\n回答:",
        "answer": "请根据以下内容回答问题，避免不必要的解释或重复。\n\n以下是参考的问题和答案风格，但生成的内容不能相同。\n\n示例问题风格: {example_question}\n示例答案风格: {example_answer}\n\n内容: {content}\n\n问题: {question}\n\n答案:",
        "answer_cot": "请根据以下提供的内容简明扼要地回答问题，然后输出一个简短的思维链 (Chain-of-Thought, CoT)，解释理解答案所需的推理或背景。输出格式: 1.答案:\n2.CoT:。避免不必要的解释或重复。\n\n内容: {content}\n\n问题: {question}\n\n1.答案:",
        "answer_style": "请根据以下内容回答问题，避免不必要的解释或重复。答案的风格应非常{style}!\n\n内容: {content}\n\n问题: {question}\n\n答案:",
        "extract": "请根据内容提取相关信息以回答问题，避免不必要的解释或重复。\n\n问题: {question}\n\n内容: {content}\n\n提取:",
        "q_a_pair": "你非常擅长从文本中提取问题和答案。基于提供的内容，构建三个直接基于文本内容的问题和答案，问题应该能够探索文本中的关键主题、角色关系、事件细节或时间顺序。每个问题和答案之间用换行符分隔。请确保问题的表达清楚地指向文本中的特定信息，避免使用模糊或过于宽泛的引用。同时，强调直接引用或文本中的具体细节，以增加问题的准确性和深度。问题的答案应当简洁明了。\n\n输出格式:问题1\n答案1\n问题2\n答案2\n...内容: {content}\n\n回答:",
        "q_a_pair_pro": "你将会收到一段来自某个专业领域的文本。基于这段文本，构建五个聚焦于该领域关键概念、术语或流程的具体问题和答案。使用任意问答风格，任意题型，如单选、多选、填空、简答等。每个问题之间用换行符分隔。问题的答案应当仅通过文本中提供的信息来作答。\n\n输出格式:问题1\n答案1\n问题2\n答案2\n...\n\n内容: {content}\n\n回答:",
        "q_a_pair_pro_select": "你将会收到一段来自某个专业领域的文本。基于这段文本，构建五个聚焦于该领域关键概念、术语或流程的具体问题和答案。题型为不定向选择题，使用每个问题之间用换行符分隔。题目后面为四个选项ABCD问题，答案直接回答选项，可以单选或多选。\n\n输出格式:问题1\n答案1\n问题2\n答案2\n...\n\n内容: {content}\n\n回答:"
    },
    "English":{
        "question":"You are a master of extracting questions from text. Based on the provided content, Construct three questions that should be directly based on the text content and able to explore key themes, character relationships, event details, or chronological order in the text. Separated by line breaks. Please ensure that the expression of the question clearly points to the specific information in the text, and avoid using vague or overly broad references. At the same time, emphasize direct references or specific details in the text to increase the accuracy and depth of the problem.\n\nDon't contain these words {filter_words}.The questions should be able to be answered in a few words or less.\n\nContent:{content}\n\nResponse:",
        "question_pro":"You will receive a passage of text from a specialized field. Based on this text, construct three specific questions that focus on key concepts, terms, or processes relevant to that field. Separated by line breaks. The questions should be answerable solely with the information provided in the text.\n\nText: {content}\n\nquestions:",
        "question_example":"You are a master of extracting questions from text. Based on the provided content, Construct three questions that should be directly based on the text content and able to explore key themes, character relationships, event details, or chronological order in the text. Separated by line breaks. Please ensure that the expression of the question clearly points to the specific information in the text, and avoid using vague or overly broad references. At the same time, emphasize direct references or specific details in the text to increase the accuracy and depth of the problem.\n\nDon't contain these words {filter_words}.The questions should be able to be answered in a few words or less.\n\nReplace pronouns with names as much as possible\n\nIf it is a relational word like kid or parents, it is necessary to indicate whose it belongs to\n\nThere are reference questions styles here, but the generated content cannot be the same.\n\nExample question style:{example_question}\n\nContent:{content}\n\nResponse:",
        "answer":"Please answer the Question according to the following Content, avoiding unnecessary explanations or repetition.\n\nThere are reference questions and answer styles here, but the generated content cannot be the same.\n\nExample question style:{example_question}\nExample answer style: {example_answer}\n\nContent:{content}\n\nQuestion:{question}\n\nAnswer:",
        "answer_cot":"Please provide a concise answer to the Question based on the Content provided below, and then output a brief Chain-of-Thought (CoT) that explains the reasoning or background necessary to understand the answer.Output format:1.Answer:\n2.CoT:.Avoid unnecessary explanations or repetition.\n\nContent: {content}\n\nQuestion: {question}\n\n1.Answer:",
        "answer_style":"Please answer the Question according to the following Content, avoiding unnecessary explanations or repetition.The style of answer should be very {style}!\n\nContent:{content}\n\nQuestion:{question}\n\nAnswer:",
        "extract":"Please extract relevant information for answering the Question based on Content to avoid unnecessary explanations or repetition\n\nQuestion:{question}\n\nContent:{content}\n\nExtract:",
        "q_a_pair":"You are a master of extracting questions and answers from text. Based on the provided content, Construct five questions and answers that should be directly based on the text content. Separated by line breaks. Please ensure that the expression of the question clearly points to the specific information in the text, and avoid using vague or overly broad references. At the same time, emphasize direct references or specific details in the text to increase the accuracy and depth of the problem.The questions should be able to be answered in a few words.\n\nOutput question and answer alternately on each line.\n\nContent:{content}\n\nResponse:"
   }
}

merge_data_prompt={
    "Chinese":{
        "prompt0":"你是一个法律知识专家，请你通过知识库直接回答问题",
        "prompt1":"你可以参考知识库中的一些片段来帮助您回答问题。参考内容：\n",
        "prompt2":"现在问题是：",
        "prompt3":"请直接回答，不要有多余的内容。",
    },
    "English":{
        "prompt0":"You are an expert who have read a lot of knowledge base. Please answer the question according to the content of the KB. ",
        "prompt1":"You can refer to some segments from the KB to help you answer the question. References: \n",
        "prompt2":"Now the question is: ",
        "prompt3":"Please answer this question.",
    }
}

long_dependency_prompt={
    "English":{
        "is_example":{
            "prompt_gen_q":"""You will receive a document, an example question and an example answer. Please refer to the example question and example answer style and output 3 generalizable, ambiguity questions (without answer), whose themes should align with the document. Separated by line breaks. 
            document:{document}
            example question:{e_q}
            example answer:{e_a}
            output:""",
            "prompt_gen_a":"""You will receive a document, an example question, an example answer and an question. Please refer to the example answer style and answer this question, if unable to answer, return 'none'; otherwise, please answer the question based on the document information.
            document:{document}
            example question:{e_q}
            example answer:{e_a}
            question:{q}
            output:
            """,
            "prompt_refine":"""You will receive an example question, an example answer, a question and an answer, where the answer is a concatenation of multiple answers. Please optimize its expression to make it smoother. Please refer to the example answer for the style of the final answer.Please output the new answer directly without any unnecessary explanation.
            example question:{e_q}
            example answer:{e_a}
            question:{q}
            answer:{a}
            output:"""
        },
        "not_example":{
            "prompt_gen_q":"""You will receive a document. Generate 3 generalizable questions based on the document content, Each question is separated by a line break.
            document: {document}
            output:""",
            "prompt_gen_a":"""You will receive a document and a question. Answer the question based on the document content. if unable to answer, return 'none'; otherwise, please answer the question based on the document information.
            document: {document}
            question: {q}
            output:""",
            "prompt_refine":"""You will receive a question and an answer, where the answer is a concatenation of multiple answers. Please optimize its expression to make it smoother.Please output the new answer directly without any unnecessary explanation.
            question: {q}
            answer: {a}
            output:"""
        }
    },
    "Chinese":{
        "not_example":{
            "prompt_gen_q":"""你将收到一份文档，请根据文档输出3个问题，每个问题以换行符隔开。
            文档:{document}
            输出：""",
            "prompt_gen_a":"""你将收到一份文档和一个问题，请根据文档回答该问题。如果无法回答问题，则返回'none'。
            文档:{document}
            问题:{q}
            输出：""",
            "prompt_refine":"""你将收到一个拼接的答案，请优化其表达，并直接输出新的答案，不要有多余的解释。
            问题:{q}
            答案:{a}
            输出："""
        }
    }
}