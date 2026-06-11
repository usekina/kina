import streamlit as st
import os
from kina import KinaAnalyzer  # 假设这是你的核心逻辑文件

# 1. 页面基本设置
st.set_page_config(page_title="Kina: Cognitive Insights", page_icon="🧠", layout="wide")

# 2. 侧边栏：关于与说明
with st.sidebar:
    st.title("Kina Analyzer")
    st.markdown("""
    **Transforming Speech into Cognitive Insights**
    
    Kina 通过分析自然对话模式来检测认知变化，助力大健康与长寿研究。
    
    [GitHub Repository](https://github.com/usekina/kina)
    """)
    st.divider()
    st.info("注：本工具仅供科研参考，不作为医疗诊断依据。")

# 3. 主界面布局
st.title("🧠 Kina: 认知洞察分析")
st.subheader("通过语音捕捉思维的细微变化")

# 创建两个并排的列
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 🎙️ 第一步：上传录音")
    uploaded_file = st.file_uploader("支持 .wav, .mp3, .m4a 格式", type=["wav", "mp3", "m4a"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # 按钮触发分析
        if st.button("🚀 开始分析认知指标", use_container_width=True):
            with st.spinner('Kina 正在提取认知特征...'):
                # --- 这里调用你的 kina.py 逻辑 ---
                # 保存临时文件供处理
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    analyzer = KinaAnalyzer()
                    results = analyzer.analyze("temp_audio.wav") # 假设你的分析函数名
                    
                    st.session_state['results'] = results
                    st.success("分析完成！")
                except Exception as e:
                    st.error(f"分析出错: {e}")

with col2:
    st.write("### 📊 第二步：分析结果")
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        # 假设你的结果包含这些维度，用进度条或指标展示
        st.metric(label="综合评分 (Cognitive Score)", value=f"{res.get('overall_score', 0)}/100")
        
        st.write("#### 关键指标维度")
        st.progress(res.get('fluency', 0) / 100, text=f"言语流畅度: {res.get('fluency', 0)}%")
        st.progress(res.get('vocabulary_richness', 0) / 100, text=f"词汇丰富度: {res.get('vocabulary_richness', 0)}%")
        st.progress(res.get('logic_coherence', 0) / 100, text=f"逻辑连贯性: {res.get('logic_coherence', 0)}%")
        
        with st.expander("查看详细建议"):
            st.write(res.get('detailed_feedback', "暂无详细反馈内容。"))
    else:
        st.info("请在左侧上传音频并点击分析。")

# 4. 底部声明
st.divider()
st.caption("© 2026 Kina Project - Built with Streamlit for CES 2026")
