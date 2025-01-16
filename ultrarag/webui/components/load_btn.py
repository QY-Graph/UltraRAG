import sys, time, asyncio
import streamlit as st
from pathlib import Path
home_path = Path().resolve()
sys.path.append(home_path.as_posix())
from ultrarag.modules.embedding import EmbeddingClient, load_model
from ultrarag.modules.llm import OpenaiLLM, HuggingfaceClient, HuggingFaceServer, VllmServer
from ultrarag.modules.reranker import BGERerankClient, BGERerankServer
from ultrarag.webui.components.loading import loading
from loguru import logger
import traceback
from transformers import AutoConfig
from ultrarag.webui.utils.language import t

@st.cache_resource
def load_embedding_model(url, device):
    logger.info(f"load embedding model from {url}, device: {device}")
    if Path(url).exists():
        return load_model(url, device)
    else:
        return EmbeddingClient(url)

@st.cache_resource
def load_llm_model(base_url, api_key, model_name, model_path, device=None, **args):
    logger.info(f"load llm model from {model_path}")
    if model_path and model_path.lstrip().startswith("http"):
        return HuggingfaceClient(url_or_path=model_path)
    if model_path and Path(model_path).exists():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # NOTE: MiniCPMV just running well in HuggingFaceServer
        if "MiniCPMV" in config.architectures:
            return HuggingFaceServer(model_path=model_path, device=device)
        return VllmServer(base_url=model_path, device=device)
    else:
        return OpenaiLLM(base_url=base_url, api_key=api_key, model=model_name)

@st.cache_resource
def load_rerank_model(url, device):
    if Path(url).exists():
        return BGERerankServer(model_path=url, device=device)
    else:
        return BGERerankClient(url=url)

@st.fragment
def update_btn(label, key, base_url=None, api_key=None, model_name=None, model_path=None):
    disabled = ((not base_url or not api_key or not model_name) and not model_path) or st.session_state.get(key + '_loaded', False)
    if st.button(
        t(label),
        disabled=disabled,
        key=key,
    ):
        st.session_state[key + '_loaded'] = True
        st.session_state["loading"] = True
        loading(t("Loading"))
        try:
            if "embedding" in key:
                if st.session_state.config['selected_devices_embedding']:
                    embed_device = st.session_state.config['selected_devices_embedding'][0]
                else:
                    embed_device = 'cpu'
                st.session_state.embedding = load_embedding_model(st.session_state.config.get('embedding_model_path'), device=embed_device) 
            elif "llm" in key:
                llm_device = st.session_state.config['selected_devices_llm']
                llm_device = llm_device[0] if llm_device else 'cpu'
                logger.info(f"llm_device: {llm_device}")
                st.session_state.llm = load_llm_model(base_url, api_key, model_name, model_path, device=llm_device)
            elif "reranker" in key:
                if st.session_state.config['selected_devices_reranker']:
                    rerank_device = st.session_state.config['selected_devices_reranker'][0]
                else:
                    rerank_device = 'cpu'
                st.session_state.reranker = load_rerank_model(st.session_state.config.get('reranker_model_path'), device=rerank_device)
        except:
            logger.error(traceback.format_exc())
            loading(t("Error"))
            time.sleep(2)
            st.session_state[key + '_loaded'] = False
            st.session_state["loading"] = False
            st.rerun()
        st.session_state["loading"] = False
        st.rerun()

def load_btn():
    
    cols = st.columns([15,1,15,1,15],vertical_alignment='bottom')
    if "llm_button_loaded" not in st.session_state:
        st.session_state.llm_button_loaded = False
    if "embedding_button_loaded" not in st.session_state:
        st.session_state.embedding_button_loaded = False
    if "reranker_button_loaded" not in st.session_state:
        st.session_state.reranker_button_loaded = False
        
    if "last_model_path" not in st.session_state:
        st.session_state.last_model_path = st.session_state.config.get('model_path')
    if "last_model_name" not in st.session_state:
        st.session_state.last_model_name = st.session_state.config.get('model_name')
    if "last_api_key" not in st.session_state:
        st.session_state.last_api_key = st.session_state.config.get('api_key')
    if "last_base_url" not in st.session_state:
        st.session_state.last_base_url = st.session_state.config.get('base_url')
    if (st.session_state.config.get('model_path') != st.session_state.last_model_path) or(
    st.session_state.config.get('model_name') != st.session_state.last_model_name) or(
    st.session_state.config.get('api_key') != st.session_state.last_api_key) or(
    st.session_state.config.get('base_url') != st.session_state.last_base_url):
        st.session_state.llm_button_loaded = False
        st.session_state.last_model_path = st.session_state.config.get('model_path')
        st.session_state.last_model_name = st.session_state.config.get('model_name')
        st.session_state.last_api_key = st.session_state.config.get('api_key')
        st.session_state.last_base_url = st.session_state.config.get('base_url')

    if "last_embedding_model_path" not in st.session_state:
        st.session_state.last_embedding_model_path = st.session_state.config.get('embedding_model_path')
    if st.session_state.config.get('embedding_model_path') != st.session_state.last_embedding_model_path:
        st.session_state.embedding_button_loaded = False
        st.session_state.last_embedding_model_path = st.session_state.config.get('embedding_model_path')

    if "last_reranker_model_path" not in st.session_state:
        st.session_state.last_reranker_model_path = st.session_state.config.get('reranker_model_path')
    if st.session_state.config.get('reranker_model_path') != st.session_state.last_reranker_model_path:
        st.session_state.reranker_button_loaded = False
        st.session_state.last_reranker_model_path = st.session_state.config.get('reranker_model_path')

    with cols[0]:
        llm_btn = update_btn(
            t("Load LLM Model"),
            'llm_button',
            base_url=st.session_state.config.get('base_url'),
            api_key=st.session_state.config.get('api_key'),
            model_name=st.session_state.config.get('model_name'),
            model_path=st.session_state.config.get('model_path'),
        )
    with cols[2]:
        embedding_btn = update_btn(
            t("Load Embedding Model"),
            'embedding_button',
            model_path=st.session_state.config.get('embedding_model_path'),
        )
    with cols[4]:
        reranker_btn = update_btn(
            t("Load Reranker Model"),
            'reranker_button',
            model_path=st.session_state.config.get('reranker_model_path'),
        )

        
        