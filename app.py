import streamlit as st
import math
import random
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="IA Anti-Fraude de Ingressos", layout="wide")

st.title("🏟️ Sistema Anti-Cambista: Rede Neural Multicamadas (MLP)")
st.markdown("""
Esta aplicação treina uma **Rede Neural Artificial do zero** para identificar fraudes e cambistas em programas de Sócio-Torcedor.
Toda a matemática (*Forward Pass*, Cálculo de Erro e *Backpropagation*) é rodada pura, sem frameworks de IA.
""")

st.sidebar.header("⚙️ Parâmetros da Rede")

# O Dataset agora tem significado real (É a lógica do XOR)
# (Volume de Compra, Frequencia na Catraca, Alerta de Fraude)
dataset = [
    (0, 0, 0), # Casual: Compra pouco, vai pouco (Liberado)
    (1, 1, 0), # Fanático: Compra muito, vai muito (Liberado)
    (1, 0, 1), # Cambista: Compra muito, não vai (Fraude!)
    (0, 1, 1)  # Invasor: Não compra, mas acessa (Fraude!)
    (0,5, 0,5, 0)    # NOVO: Torcedor Comum: Compra médio, vai médio (Liberado)
]

st.sidebar.markdown("**Regras de Treinamento (Dataset):**")
st.sidebar.markdown("- (0,0) -> 0 (Casual)")
st.sidebar.markdown("- (1,1) -> 0 (Fanático)")
st.sidebar.markdown("- (1,0) -> 1 (Cambista)")
st.sidebar.markdown("- (0,1) -> 1 (Invasor)")

# Hiperparâmetros
lr = st.sidebar.slider("Taxa de Aprendizado (Learning Rate):", 0.01, 2.0, 0.8, step=0.05)
epochs = st.sidebar.slider("Número de Épocas:", 1000, 30000, 15000, step=1000)
seed = st.sidebar.number_input("Semente Aleatória (Seed):", value=1, step=1)

# Função de ativação Sigmoide
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Botão de Treinamento
if st.button("🚀 Treinar Rede Anti-Fraude"):
    random.seed(seed)
    
    # Pesos Iniciais
    w11, w12, b1 = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
    w21, w22, b2 = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
    v1, v2, c = random.uniform(-1, 1), random.uniform(-1, 1), 0.0

    def forward(x1, x2):
        h1 = sigmoid(w11 * x1 + w12 * x2 + b1)
        h2 = sigmoid(w21 * x1 + w22 * x2 + b2)
        o = sigmoid(v1 * h1 + v2 * h2 + c)
        return h1, h2, o

    def calcular_mse():
        total = sum((t - forward(x1, x2)[2]) ** 2 for x1, x2, t in dataset)
        return total / len(dataset)

    historico_epocas = []
    historico_perda = []

    # Backpropagation puro
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        for x1, x2, t in dataset:
            h1, h2, o = forward(x1, x2)
            erro = t - o
            d_o = erro * o * (1 - o)
            
            v1 += lr * d_o * h1
            v2 += lr * d_o * h2
            c += lr * d_o
            
            d_h1 = d_o * v1 * h1 * (1 - h1)
            w11 += lr * d_h1 * x1
            w12 += lr * d_h1 * x2
            b1 += lr * d_h1
            
            d_h2 = d_o * v2 * h2 * (1 - h2)
            w21 += lr * d_h2 * x1
            w22 += lr * d_h2 * x2
            b2 += lr * d_h2

        if epoch % (epochs // 50) == 0 or epoch == epochs - 1:
            historico_epocas.append(epoch)
            historico_perda.append(calcular_mse())
            progress_bar.progress((epoch + 1) / epochs)

    st.session_state['treinado'] = True
    st.session_state['modelo'] = (w11, w12, b1, w21, w22, b2, v1, v2, c)
    st.session_state['historico'] = (historico_epocas, historico_perda)

# Exibição
if st.session_state.get('treinado', False):
    w11, w12, b1, w21, w22, b2, v1, v2, c = st.session_state['modelo']
    historico_epocas, historico_perda = st.session_state['historico']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Gráfico de Aprendizado (Queda de Erro)")
        fig, ax = plt.subplots()
        ax.plot(historico_epocas, historico_perda, color="#d62728", linewidth=2)
        ax.set_xlabel("Épocas de Treino")
        ax.set_ylabel("Erro (MSE)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.subheader("🎯 Teste do Perfil do Usuário")
        st.write("Ajuste o comportamento do cliente:")
        
        # Teste ao vivo
        input_x1 = st.slider("Volume de Compras no Site (0=Pouco, 1=Muito):", 0.0, 1.0, 0.5, step=0.05)
        input_x2 = st.slider("Presença na Catraca (0=Não vai, 1=Sempre vai):", 0.0, 1.0, 0.5, step=0.05)
        
        h1 = sigmoid(w11 * input_x1 + w12 * input_x2 + b1)
        h2 = sigmoid(w21 * input_x1 + w22 * input_x2 + b2)
        pred_o = sigmoid(v1 * h1 + v2 * h2 + c)
        
        st.metric(label="Risco de Fraude (0.0 a 1.0)", value=f"{pred_o:.4f}")
        
        if pred_o >= 0.5:
            st.error("🚨 CONTA BLOQUEADA - ALTO RISCO DE FRAUDE/CAMBISMO")
        else:
            st.success("✅ ACESSO LIBERADO - TORCEDOR GENUÍNO")
