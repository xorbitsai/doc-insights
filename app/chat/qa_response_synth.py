from llama_index import PromptTemplate
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompt_type import PromptType
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.response_synthesizers import BaseSynthesizer
from llama_index.response_synthesizers.factory import get_response_synthesizer

from .utils import build_title_for_document


def get_custom_response_synthesizer(
    service_context: ServiceContext, documents
) -> BaseSynthesizer:
    doc_titles = "\n".join("- " + build_title_for_document(doc) for doc in documents)

    refine_template_str = f"""
用户选择了一组基金文档，并询问了有关这些文件的问题。这些基金文件的标题如下:
{doc_titles}
原始查询如下:{{query_str}}
我们已经提供了一个答案:{{existing_answer}}
如果需要请改善现有的答案 ，以下有更多的背景信息。
------------
{{context_msg}}
------------
考虑到新的上下文，将原来的答案改进为更好的答案回答这个问题。如果上下文没有用，则返回原始答案。
完善后的回答:
""".strip()
    refine_prompt = RefinePrompt(
        template=refine_template_str, prompt_type=PromptType.REFINE
    )

    qa_template_str = f"""
用户选择了一组基金文档，并询问了有关这些文件的问题。基金文件的标题如下:
{doc_titles}
下面是上下文信息。
---------------------
{{context_str}}
---------------------
给定上下文信息而不是先验知识，回答这个问题，总是使用中文回复。
问题: {{query_str}}
回答:
""".strip()
    qa_prompt = QuestionAnswerPrompt(
        template=qa_template_str,
        prompt_type=PromptType.QUESTION_ANSWER,
    )

    return get_response_synthesizer(
        service_context, refine_template=refine_prompt, text_qa_template=qa_prompt
    )


def get_template():
    return PromptTemplate(
        """
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""".strip()
    )
