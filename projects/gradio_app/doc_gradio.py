import base64
import os
import re
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import pymupdf
from gradio_pdf import PDF
from loguru import logger

from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.tools.common import do_parse, prepare_env

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# For RAG functionality
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 加载 Hugging Face 的 ChineseErrorCorrector2_7b_DoCurement 模型
tokenizer = AutoTokenizer.from_pretrained("HMonKY/ChineseErrorCorrector2_7b_DoCurement", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("HMonKY/ChineseErrorCorrector2_7b_DoCurement", trust_remote_code=True)

# 加载嵌入模型用于RAG
embedding_model = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

# 全局变量存储当前文档的向量数据库
current_vectordb = None


def read_fn(path):
    disk_rw = FileBasedDataReader(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path))


def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, layout_mode, formula_enable, table_enable, language):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{str(Path(doc_path).stem)}_{time.time()}'
        pdf_data = read_fn(doc_path)
        if is_ocr:
            parse_method = 'ocr'
        else:
            parse_method = 'auto'
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
            layout_model=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            lang=language,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。"""
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def to_markdown(file_path, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
    file_path = to_pdf(file_path)
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1, is_ocr,
                                        layout_mode, formula_enable, table_enable, language)
    archive_zip_path = os.path.join('./output', compute_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('压缩成功')
    else:
        logger.error('压缩失败')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')

    return md_content, txt_content, archive_zip_path, new_pdf_path


latex_delimiters = [{'left': '$$', 'right': '$$', 'display': True},
                    {'left': '$', 'right': '$', 'display': False}]


def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)  # noqa: F841
        logger.info('txt_model init final')
        ocr_model = model_manager.get_model(True, False)  # noqa: F841
        logger.info('ocr_model init final')
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


model_init = init_model()
logger.info(f'model_init: {model_init}')


with open('header.html', 'r', encoding='utf-8') as file:
    header = file.read()


latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',  
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',  
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',  
        'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']
add_lang = ['latin', 'arabic', 'cyrillic', 'devanagari']

all_lang = []
all_lang.extend([*other_lang, *add_lang])


def to_pdf(file_path):
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            # 生成唯一的文件名
            unique_filename = f'{uuid.uuid4()}.pdf'
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)
            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)
            return tmp_file_path


def grammar_check(text, role="user"):
    # 使用更简洁的提示词格式
    prompt = f"""作为文档分析助手，请分析以下文本中的语法错误并提供修改建议（没有错误不要硬写），再分析文本风格及修改意见：
{text}
请直接给出分析结果，简洁明了，不要重复内容："""
    
    # 增加输入的最大长度
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    
    # 获取prompt的token数量
    prompt_length = inputs["input_ids"].shape[1]
    
    # 添加多种控制重复的参数
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        # 添加以下参数控制重复
        do_sample=True,  # 启用采样
        temperature=0.7,  # 控制随机性 (0.7是个好的平衡点)
        top_p=0.9,        # 使用nucleus sampling
        repetition_penalty=1.2,  # 惩罚重复
        no_repeat_ngram_size=3,  # 禁止重复的n-gram大小
    )
    
    # 只解码模型生成的新token部分
    generated_tokens = outputs[0][prompt_length:]
    report = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 额外处理：如果仍有重复，尝试只保留第一段有意义的内容
    sentences = report.split('。')
    if len(sentences) > 2:
        # 检查是否有重复句子
        unique_sentences = []
        for sentence in sentences:
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # 重建报告，只使用唯一的句子
        report = '。'.join(unique_sentences)
        if not report.endswith('。') and len(report) > 0:
            report += '。'
    
    return report


def export_report(report_text, export_format):
    """
    导出 report 内容到文件。export_format 参数可以为 "txt" 或 "pdf"。
    """
    os.makedirs("export_reports", exist_ok=True)
    filename = f"grammar_report_{int(time.time())}.{export_format}"
    file_path = os.path.join("export_reports", filename)
    if export_format == "txt":
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    elif export_format == "pdf":
        try:
            from fpdf import FPDF
        except ImportError:
            logger.error("fpdf 库未安装，请先安装：pip install fpdf")
            return None
        pdf = FPDF()
        pdf.add_page()
        try:
            # 尝试添加支持 Unicode 的 DejaVuSans 字体
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', size=12)
        except Exception as e:
            logger.error("无法加载Unicode字体 DejaVuSans.ttf，请确保字体文件存在。错误信息：" + str(e))
            # 如果加载失败，回退到默认字体（可能无法完整显示中文）
            pdf.set_font("Arial", size=12)

        for line in report_text.splitlines():
            pdf.cell(200, 10, txt=line, ln=1)
        pdf.output(file_path)
    return file_path


# 新增：创建文档的向量数据库用于RAG
def create_vectordb(text):
    global current_vectordb
    
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    # 分割文本
    chunks = text_splitter.split_text(text)
    
    # 创建向量数据库
    current_vectordb = FAISS.from_texts(chunks, embedding_model)
    
    return "Updated! Well done!"


# 新增：基于RAG的问答功能
def rag_qa(question):
    global current_vectordb
    
    if current_vectordb is None:
        return "Please upload text first."
    
    # 从向量数据库检索相关内容
    docs = current_vectordb.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 构建提示词
    prompt = f"""请基于以下参考信息回答用户的问题。如果参考信息中没有相关内容，请如实说明无法回答。

参考信息:
{context}

用户问题: {question}

请给出简洁明了的回答:"""
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    prompt_length = inputs["input_ids"].shape[1]
    
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    
    # 解码生成的回答
    generated_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return answer


def add_grammar_module(demo):
    with gr.Column(variant='panel', scale=2):
        grammar_input = gr.TextArea(
            label="Text for Analysis",
            placeholder="Enter text here...",
            lines=5,
            max_lines=10
        )
        
        grammar_report_button = gr.Button("Generate Report")
        # 将生成的report框设为可编辑
        grammar_report_output = gr.TextArea(
            label="Report", 
            placeholder="The report will be displayed here...",
            lines=10,
            interactive=True  # 改成 True，用户可以编辑
        )
        # 下拉菜单，选择导出格式
        export_format_dropdown = gr.Dropdown(
            label="Export Format", 
            choices=["txt", "pdf"],
            value="txt"
        )
        # "一键导出"按钮
        export_button = gr.Button("Export")
        # 用于显示导出的文件链接，类型为File
        export_file_output = gr.File(label="Export Files", interactive=False)
        
        # 生成报告的按钮点击事件
        grammar_report_button.click(
            fn=grammar_check, 
            inputs=[grammar_input], 
            outputs=[grammar_report_output]
        )
        # 导出按钮点击事件，输入当前 report 的文本与导出格式，并生成文件
        export_button.click(
            fn=export_report,
            inputs=[grammar_report_output, export_format_dropdown],
            outputs=[export_file_output]
        )
    return demo, grammar_input  # 返回grammar_input以便后续使用


# 添加智能问答模块
def add_qa_module(demo):
    with gr.Column(variant='panel', scale=2):
        gr.Markdown("### Q&A")
        
        # 更新向量数据库按钮
        update_vectordb_button = gr.Button("Update Knowledge")
        update_status = gr.Textbox(label="state", interactive=False)
        
        # 问答界面
        user_question = gr.TextArea(
            label="Questions",
            placeholder="Please write down your questions...",
            lines=2
        )
        
        answer_button = gr.Button("Get Answers")
        
        answer_output = gr.TextArea(
            label="Answers", 
            placeholder="The answers will be displayed here...",
            lines=8,
            interactive=False
        )
        
    return demo, update_vectordb_button, update_status, user_question, answer_button, answer_output


# Gradio 界面
with gr.Blocks() as demo:
    gr.HTML(header)
    with gr.Row():
        with gr.Column(variant='panel', scale=5):
            file = gr.File(label='Please upload a PDF or image', file_types=['.pdf', '.png', '.jpeg', '.jpg'])
            max_pages = gr.Slider(1, 20, 10, step=1, label='Max convert pages')
            with gr.Row():
                layout_mode = gr.Dropdown(['doclayout_yolo'], label='Layout model', value='doclayout_yolo')
                language = gr.Dropdown(all_lang, label='Language', value='ch')
            with gr.Row():
                formula_enable = gr.Checkbox(label='Enable formula recognition', value=True)
                is_ocr = gr.Checkbox(label='Force enable OCR', value=False)
                table_enable = gr.Checkbox(label='Enable table recognition(test)', value=True)
            with gr.Row():
                change_bu = gr.Button('Convert')
                clear_bu = gr.ClearButton(value='Clear')
            pdf_show = PDF(label='PDF preview', interactive=False, visible=True, height=800)
            with gr.Accordion('Examples:'):
                example_root = os.path.join(os.path.dirname(__file__), 'examples')
                gr.Examples(
                    examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if _.endswith('pdf')],
                    inputs=file
                )

        with gr.Column(variant='panel', scale=5):
            output_file = gr.File(label='convert result', interactive=False)
            with gr.Tabs():
              with gr.Tab('Markdown rendering'):
                md = gr.Markdown(label='Markdown rendering', height=1100, show_copy_button=True,
                             latex_delimiters=latex_delimiters, line_breaks=True)
              with gr.Tab('Markdown text'):
                # 将md_text改为可编辑的
                md_text = gr.TextArea(lines=45, show_copy_button=True, interactive=True)
                with gr.Row():
                  # 添加更新Markdown预览的按钮
                  update_preview_button = gr.Button("Update Preview")
                  # 添加"Finished"按钮
                  finished_button = gr.Button("Finished")

    with gr.Row():
        # 语法检查模块
        demo, grammar_input = add_grammar_module(demo)
        # 智能问答模块
        demo, update_vectordb_button, update_status, user_question, answer_button, answer_output = add_qa_module(demo)

    # 文件上传和转换按钮事件
    file.change(fn=to_pdf, inputs=file, outputs=pdf_show)
    change_bu.click(fn=to_markdown,
                    inputs=[file, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
                    outputs=[md, md_text, output_file, pdf_show])
    clear_bu.add([file, md, pdf_show, md_text, output_file, is_ocr])
    
    # 更新预览按钮的点击事件
    update_preview_button.click(
        fn=lambda text: text,  # 简单地传递文本
        inputs=[md_text],
        outputs=[md]  # 将编辑后的文本更新到Markdown渲染区域
    )

    # 添加Finished按钮的点击事件，将编辑后的md_text传递到grammar_input
    finished_button.click(
        fn=lambda text: text,  # 简单地传递文本
        inputs=[md_text],
        outputs=[grammar_input]
    )
    
    # 添加更新向量数据库按钮的点击事件
    update_vectordb_button.click(
        fn=create_vectordb,
        inputs=[md_text],
        outputs=[update_status]
    )
    
    # 添加问答按钮的点击事件
    answer_button.click(
        fn=rag_qa,
        inputs=[user_question],
        outputs=[answer_output]
    )

demo.launch(server_name='0.0.0.0', share=True)

